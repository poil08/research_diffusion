#!/usr/bin/env python3
"""Step 3: full selection-only Fourier cutoff sweep for v3 1000/1000 preparation.
Standalone CPU script. No evaluation tensors, no model inference, no downloads.
Saves resumable chunks, all scores, frozen choices and static report figures.
"""
import argparse
import csv
import os
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time

N_IMAGES = 1000
RADII = [.025,.05,.075,.1,.125,.15,.2,.25,.3,.4,.5,None]
TIMESTEPS = list(range(0,1000,10))+[999]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dependencies(threads=1):
    global np, torch, Image, cdist
    import numpy as np
    import torch
    from PIL import Image
    from scipy.spatial.distance import cdist
    torch.set_num_threads(threads)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(x):
    a=x.detach().cpu().contiguous().numpy()
    h=hashlib.sha256(str((str(a.dtype),a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def noise(seeds):
    return torch.stack([torch.randn((3,64,64),generator=torch.Generator(device='cpu').manual_seed(s),
                                    dtype=torch.float32,device='cpu') for s in seeds])


def noisy(clean, eps, alpha, t):
    a=alpha[t]
    return a.sqrt()*clean+(1-a).sqrt()*eps


class Preparation:
    """Only explicit shared metadata and selection/clean.pt can be opened."""
    ALLOWED={'manifest.json','noise_plan.json','planned_protocol.json','split.json',
             'selected_images.json','schedule.pt','selection/clean.pt'}

    def __init__(self, root):
        self.root=Path(root).resolve(strict=True)
        self.read_files=[]
        self.input_hashes={}
        self.m=self.read_json('manifest.json',verify=False)
        require(self.m.get('schema_version')==3 and self.m.get('status')=='complete'
                and self.m.get('experiment')=='cifar10_cutoff_lazy_preparation',
                'Expected a completed v3 lazy-preparation folder')
        records=self.m.get('artifacts',[])
        names=[r['file'] for r in records]
        require(len(names)==len(set(names)), 'Duplicate artifact entries')
        self.artifacts={r['file']:r['sha256'] for r in records}
        require(self.m['versions']['torch']==importlib.metadata.version('torch'),
                'Torch version differs from preparation; use the same environment (no version bypass)')
        require(self.m['dataset_split']=='train' and self.m['timesteps']==TIMESTEPS
                and self.m['repetitions']==5, 'Unexpected split/timesteps/repetitions')
        require(self.m['preprocessing']=={'size':[64,64],'resize':'Pillow bilinear',
                'normalization':'uint8 / 127.5 - 1','random_augmentation':False},
                'Preprocessing differs from the agreed protocol')
        require(self.m['noise']['algorithm']=='per_image_cpu_torch_randn_v1'
                and self.m['noise']['device']=='cpu' and self.m['noise']['dtype']=='float32',
                'Unsupported noise reconstruction recipe')
        self.protocol=self.read_json('planned_protocol.json')
        self.validate_protocol()
        splits=self.read_json('split.json')
        all_records=self.read_json('selected_images.json')
        ids=[]
        for group in ('selection','evaluation'):
            g=self.m['groups'][group]
            rows=g['records']
            require(g['n_images']==N_IMAGES and len(rows)==N_IMAGES,'Unexpected group size')
            require([sum(r['label']==k for r in rows) for k in range(10)]==[N_IMAGES//10]*10,
                    'Incorrect class balance')
            group_ids=[r['sample_id'] for r in rows]
            require(splits[group]==group_ids, 'Split order differs from manifest')
            for r in rows:
                require(type(r['dataset_index']) is int and 0<=r['dataset_index']<50000
                        and r['sample_id']==f"train_{r['dataset_index']:05d}"
                        and r['group']==group, 'Invalid image identity')
            ids.extend(group_ids)
        require(all_records==self.m['groups']['selection']['records']+self.m['groups']['evaluation']['records'],
                'Combined identity list differs')
        old=self.m['previous_preparation']['excluded_sample_ids']
        require(len(old)==len(set(old))==100 and len(ids)==len(set(ids))==2*N_IMAGES
                and not set(ids).intersection(old), 'Group overlap or previous-ID exclusion failed')
        payload=self.read_tensor('selection/clean.pt')
        self.clean=payload['samples']
        original=payload['original_uint8']
        group=self.m['groups']['selection']
        require(payload['group']=='selection' and payload['records']==group['records']
                and payload['sample_ids']==splits['selection'], 'Selection tensor ID/order mismatch')
        require(self.clean.dtype==torch.float32 and tuple(self.clean.shape)==(N_IMAGES,3,64,64)
                and torch.isfinite(self.clean).all() and self.clean.min()>=-1 and self.clean.max()<=1,
                'Invalid clean tensor')
        require(original.dtype==torch.uint8 and tuple(original.shape)==(N_IMAGES,3,32,32),
                'Invalid original tensor')
        require(tensor_hash(self.clean)==group['clean_tensor_sha256']
                and tensor_hash(original)==group['original_tensor_sha256'],'Selection tensor hash mismatch')
        seen=set()
        for i in range(N_IMAGES):
            im=Image.fromarray(original[i].permute(1,2,0).numpy())
            arr=np.array(im.resize((64,64),Image.Resampling.BILINEAR),copy=True)
            expected=torch.from_numpy(arr).permute(2,0,1).float().div(127.5).sub(1)
            require(torch.equal(expected,self.clean[i]), 'Resizing/normalization reconstruction mismatch')
            h=tensor_hash(self.clean[i])
            require(h not in seen, 'Duplicate prepared selection image')
            seen.add(h)
        schedule=self.read_tensor('schedule.pt')
        betas,self.alpha=schedule['betas'],schedule['alphas_cumprod']
        require(betas.dtype==self.alpha.dtype==torch.float32 and betas.shape==self.alpha.shape==(1000,),
                'Invalid scheduler tensor shapes/dtypes')
        require(torch.isfinite(betas).all() and ((betas>0)&(betas<1)).all()
                and torch.isfinite(self.alpha).all() and ((self.alpha>0)&(self.alpha<1)).all()
                and (self.alpha[1:]<=self.alpha[:-1]).all()
                and torch.allclose(self.alpha,torch.cumprod(1-betas,0),atol=0,rtol=1e-6),
                'Invalid beta/alpha schedule')
        for src in self.m['scheduler_sources'].values():
            require(tensor_hash(betas)==src['betas_sha256'],'Saved beta hash differs from model configuration record')
        plan=self.read_json('noise_plan.json')
        require(len(plan)==20, 'Expected 20 noise bank plans')
        self.rows={}
        seeds=set()
        for row in plan:
            group,rep,bank=row['group'],row['repetition'],row['bank']
            require(group in ('selection','evaluation') and type(rep) is int and 0<=rep<5
                    and bank in ('A','B'), 'Invalid noise stream identity')
            key=(group,rep,bank)
            require(key not in self.rows, 'Duplicate noise stream')
            self.rows[key]=row
            values=row['seeds_in_saved_sample_order']
            stream=((0 if group=='selection' else 1)*5+rep)*2+('A','B').index(bank)
            base=self.m['arguments']['noise_seed_base']+stream*N_IMAGES
            require(values==list(range(base,base+N_IMAGES)) and all(type(s) is int and 0<=s<2**63 for s in values),
                    'Noise seeds differ from preparation recipe')
            require(not seeds.intersection(values), 'Overlapping noise seeds')
            seeds.update(values)
        require(not seeds.intersection(self.m['previous_preparation']['previous_noise_seeds']),
                'Noise seeds overlap with prior experiment')
        self.noise_checks=[]
        hashes=set()
        for rep in range(5):
            for bank in ('A','B'):
                row=self.rows[('selection',rep,bank)]
                eps=noise(row['seeds_in_saved_sample_order'])
                h=tensor_hash(eps)
                require(h==row['noise_tensor_sha256'] and h not in hashes, 'Noise bank hash mismatch/duplicate')
                hashes.add(h)
                ix=row['probe_positions']
                require(ix==sorted(set([0,N_IMAGES//2,N_IMAGES-1])), 'Unexpected probe positions')
                require(torch.equal(eps[ix],noise([row['seeds_in_saved_sample_order'][i] for i in ix])),
                        'Noise depends on batch composition')
                require(set(row['probe_state_sha256'])=={'0','500','999'}, 'Missing state probe hashes')
                for t,digest in row['probe_state_sha256'].items():
                    state=noisy(self.clean[ix],eps[ix],self.alpha,int(t))
                    require(tensor_hash(state)==digest, 'Reconstructed state hash mismatch')
                self.noise_checks.append({'repetition':rep,'bank':bank,'sha256':h})
                del eps
            print(f'Validated selection noise pair {rep+1}/5',flush=True)

    def path(self,name,verify=True):
        require(name in self.ALLOWED, f'Refusing non-allowlisted input: {name}')
        path=self.root/name
        require(not any(p.is_symlink() for p in [path,*path.parents] if p!=self.root.parent),
                'Symlinked input paths are not supported')
        require(path.resolve().is_relative_to(self.root) and path.is_file(), 'Invalid input file')
        digest=file_hash(path)
        if verify:
            require(name in self.artifacts and digest==self.artifacts[name],f'Artifact hash mismatch: {name}')
        self.input_hashes[name]=digest
        self.read_files.append(name)
        return path

    def read_json(self,name,verify=True):
        return json.loads(self.path(name,verify).read_text())

    def read_tensor(self,name):
        return torch.load(self.path(name),map_location='cpu',weights_only=True)

    def validate_protocol(self):
        p=self.protocol
        require(p['status']=='prespecified_not_run' and p['candidate_radii']==RADII
                and p['n_candidates_per_query']==N_IMAGES and p['chance_reference']==1/N_IMAGES,
                'Candidate/protocol size mismatch')
        require(p['feature']=='grayscale(.299,.587,.114); subtract spatial mean; orthonormal FFT2; log1p(abs(FFT)**2); discard phase; exclude DC'
                and p['distance']=='RMS feature difference over included bins'
                and p['mask']=='0 < sqrt(fx**2+fy**2) <= R', 'Unsupported FFT/distance protocol')
        require(p['distance_ties']=={'atol':1e-10,'rtol':1e-7,'credit':'1/k if correct among k nearest tied candidates'}
                and p['selection']=={'rule':'maximize mean accuracy at each t; smallest R wins ties; full last',
                                     'tie_atol':1e-12,'tie_rtol':0,'smoothing':'none'}, 'Tie/selection rule mismatch')
        require(p['evaluation_methods']==['frozen_selected_R(t)','fixed_R=0.1','full_spectrum']
                and p['controls']==['clean vs clean','independent pure-noise A/B'], 'Baseline/control mismatch')


def masks_and_specs():
    f=np.fft.fftfreq(64)
    radius=np.hypot(f[:,None],f[None,:])
    masks=[(radius>0) if r is None else ((radius>0)&(radius<=r)) for r in RADII]
    specs=[{'radius':r,'name':'full' if r is None else f'R={r:g}','n_bins':int(m.sum()),
            'mask_sha256':hashlib.sha256(m.tobytes()).hexdigest()} for r,m in zip(RADII,masks)]
    require([s['n_bins'] for s in specs]==[8,36,68,128,196,292,508,796,1152,2060,3206,4095],
            'Unexpected 64x64 frequency masks')
    return masks,specs


def fft_features(x,batch_size=64):
    result=np.empty((len(x),64,64),dtype=np.float64)
    for start in range(0,len(x),batch_size):
        rgb=x[start:start+batch_size].cpu().numpy().astype(np.float64)
        gray=np.einsum('nchw,c->nhw',rgb,[.299,.587,.114])
        gray-=gray.mean(axis=(-2,-1),keepdims=True)
        fft=np.fft.fft2(gray,axes=(-2,-1),norm='ortho')
        fft[:,0,0]=0
        result[start:start+batch_size]=np.log1p(np.abs(fft)**2)
    require(np.isfinite(result).all(),'Nonfinite FFT features')
    return result


def distance_sweep(a,b,masks,batch_size):
    """Exact float64 annulus accumulation; one NxN matrix, row-batched cdist."""
    total=np.zeros((len(a),len(b)),dtype=np.float64)
    previous=np.zeros_like(masks[0])
    for mask in masks:
        require(not (previous & ~mask).any(),'Non-nested masks')
        extra=mask & ~previous
        aa,bb=a[:,extra],b[:,extra]
        for start in range(0,len(a),batch_size):
            total[start:start+batch_size]+=cdist(aa[start:start+batch_size],bb,metric='sqeuclidean')
        yield np.sqrt(total/int(mask.sum()))
        previous=mask


def credits(d):
    require(d.ndim==2 and d.shape[0]==d.shape[1] and np.isfinite(d).all(),'Invalid distance matrix')
    tied=np.isclose(d,d.min(axis=1,keepdims=True),atol=1e-10,rtol=1e-7)
    return np.diag(tied)/tied.sum(axis=1),tied.sum(axis=1)


def numerical_checks(source,masks,batch_size):
    n=min(16,len(source.clean))
    clean=source.clean[:n]
    ea=noise(source.rows[('selection',0,'A')]['seeds_in_saved_sample_order'][:n])
    eb=noise(source.rows[('selection',0,'B')]['seeds_in_saved_sample_order'][:n])
    errors=[]
    feature_errors=[]
    repeat_errors=[]
    for t in (0,500,999):
        a=fft_features(noisy(clean,ea,source.alpha,t),batch_size)
        b=fft_features(noisy(clean,eb,source.alpha,t),batch_size)
        reference_features=fft_features(noisy(clean,ea,source.alpha,t),1)
        # NumPy reductions may differ at floating-point roundoff across batch layouts.
        require(np.allclose(a,reference_features,atol=1e-12,rtol=1e-12), 'FFT batch invariance failed')
        feature_errors.append(float(np.max(np.abs(a-reference_features))))
        repeated=fft_features(noisy(clean,ea,source.alpha,t),batch_size)
        require(np.allclose(a,repeated,atol=1e-12,rtol=1e-12),
                'Same-batch FFT repeatability failed')
        repeat_errors.append(float(np.max(np.abs(a-repeated))))
        for mask,d in zip(masks,distance_sweep(a,b,masks,batch_size)):
            # Independent direct subtraction on small probe, not cumulative cdist.
            direct=np.sqrt(np.mean((a[:,mask][:,None,:]-b[:,mask][None,:,:])**2,axis=-1))
            require(np.allclose(d,direct,atol=1e-12,rtol=1e-12),'Direct vs batched distances disagree')
            for mat,ref in ((d,direct),(d.T,direct.T)):
                require(np.array_equal(credits(mat)[0],credits(ref)[0]),'Retrieval credit differs from direct calculation')
            errors.append(float(np.max(np.abs(d-direct))))
        reference_b=fft_features(noisy(clean,eb,source.alpha,t),1)
        require(np.allclose(b,reference_b,atol=1e-12,rtol=1e-12),'Bank B FFT batch invariance failed')
        feature_errors.append(float(np.max(np.abs(b-reference_b))))
        for x,y in zip(distance_sweep(a,b,masks,batch_size),distance_sweep(reference_features,reference_b,masks,1)):
            require(np.allclose(x,y,atol=1e-12,rtol=1e-12),'Distance results depend on batch size')
            require(np.array_equal(credits(x)[0],credits(y)[0])
                    and np.array_equal(credits(x.T)[0],credits(y.T)[0]),'Batch size changes retrieval credit')
    c=fft_features(clean,batch_size)
    controls=[]
    for spec,d in zip(masks_and_specs()[1],distance_sweep(c,c,masks,batch_size)):
        require(np.array_equal(np.diag(d),np.zeros(n)) and np.array_equal(d,d.T),
                'Clean identity distances not zero/symmetric')
        credit,ties=credits(d)
        # FFT magnitudes can legitimately tie for different originals: report, do not force 100%.
        controls.append({'candidate':spec['name'],'mean_accuracy':float(credit.mean()),
                         'n_tied_queries':int((ties>1).sum()),'n_candidates':n})
    tie_test=np.array([[0.,0.],[0.,0.]])
    require(np.array_equal(credits(tie_test)[0],[.5,.5]),'Fractional tie handling failed')
    return {'probe_images':n,'probe_timesteps':[0,500,999],
            'max_distance_error_vs_direct':max(errors),'batch_invariance_passed':True,
            'max_fft_error_across_batch_sizes':max(feature_errors),
            'max_fft_error_same_batch_repeat':max(repeat_errors),
            'fft_batch_tolerance':{'atol':1e-12,'rtol':1e-12},
            'bidirectional_credit_agreement_passed':True,'fractional_tie_check_passed':True,
            'clean_probe_controls':controls}



def atomic_json(path,obj):
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp')
    write_json(tmp,obj)
    os.replace(tmp,path)


def save_csv(path,rows):
    require(bool(rows),'No rows to save')
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_validation(path,source):
    path=Path(path).resolve(strict=True)
    report=json.loads(path.read_text())
    require(report.get('status')=='complete'
            and report.get('experiment')=='selection_only_v3_validation_and_timing'
            and report.get('evaluation_tensors_read') is False
            and report.get('cutoff_search_performed') is False,
            'Expected the completed Script 2 validation report')
    require(report['input_sha256']==source.input_hashes,'Validation used different preparation inputs')
    for name in ('torch','numpy','scipy','Pillow'):
        require(report['versions'][name]==importlib.metadata.version(name),
                f'{name} version differs from validated environment')
    checks=report['numerical_checks']
    require(checks['batch_invariance_passed'] and checks['bidirectional_credit_agreement_passed']
            and checks['fractional_tie_check_passed'], 'Script 2 numerical checks not passed')
    return {'path':str(path),'sha256':file_hash(path)}


def score_pair(a,b,masks,batch_size):
    results,ties=[],[]
    for d in distance_sweep(a,b,masks,batch_size):
        ab,na=credits(d)
        ba,nb=credits(d.T)
        results.append(np.stack([ab,ba]))
        ties.append(np.stack([na>1,nb>1]))
    return np.stack(results),np.stack(ties)


def save_chunk(out,key,result,tied,state):
    folder=out/'chunks'
    path=folder/(key+'.npz')
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,credit=result,tied=tied,key=np.array(key))
    os.replace(tmp,path)
    state['chunks'][key]={'file':str(path.relative_to(out)),'sha256':file_hash(path)}
    state['updated_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(out/'analysis_manifest.json',state)


def read_chunk(out,key,state):
    entry=state['chunks'][key]
    require(entry['file']==f'chunks/{key}.npz','Unexpected chunk path')
    path=out/entry['file']
    require(not path.is_symlink() and path.is_file() and file_hash(path)==entry['sha256'],
            f'Chunk missing or changed: {key}')
    with np.load(path,allow_pickle=False) as f:
        require(str(f['key'])==key,'Chunk identity mismatch')
        credit,tied=f['credit'].copy(),f['tied'].copy()
    require(credit.shape==(12,2,N_IMAGES) and tied.shape==credit.shape
            and tied.dtype==np.bool_ and credit.dtype==np.float64
            and np.isfinite(credit).all() and ((credit>=0)&(credit<=1)).all(),
            'Invalid saved query credits')
    return credit,tied


def choose_best(means):
    winners=np.flatnonzero(np.isclose(means,means.max(),atol=1e-12,rtol=0))
    return int(winners[0]),winners.tolist()


def summarize(out,state,specs):
    scores=np.empty((5,len(TIMESTEPS),12),dtype=np.float64)
    tie_rates=np.empty_like(scores)
    repeated=[]
    for rep in range(5):
        for ti,t in enumerate(TIMESTEPS):
            q,tied=read_chunk(out,f'rep_{rep:02d}_t_{t:04d}',state)
            scores[rep,ti]=q.mean(axis=(1,2))
            tie_rates[rep,ti]=tied.mean(axis=(1,2))
            for ci,spec in enumerate(specs):
                repeated.append({'repetition':rep,'timestep':t,'candidate':spec['name'],
                                 'radius':spec['radius'],'accuracy':float(scores[rep,ti,ci]),
                                 'accuracy_a_to_b':float(q[ci,0].mean()),
                                 'accuracy_b_to_a':float(q[ci,1].mean()),
                                 'tied_query_fraction':float(tie_rates[rep,ti,ci])})
    summary,choices=[],[]
    for ti,t in enumerate(TIMESTEPS):
        mean=scores[:,ti].mean(axis=0)
        best,tied=choose_best(mean)
        for ci,spec in enumerate(specs):
            values=scores[:,ti,ci]
            summary.append({'timestep':t,'candidate':spec['name'],'radius':spec['radius'],
                            'n_bins':spec['n_bins'],'mean_accuracy':float(mean[ci]),
                            'min_repeat_accuracy':float(values.min()),
                            'max_repeat_accuracy':float(values.max()),
                            'mean_tied_query_fraction':float(tie_rates[:,ti,ci].mean()),
                            'is_selected':ci==best,'is_tied_best':ci in tied})
        choices.append({'timestep':t,'candidate':specs[best]['name'],'radius':specs[best]['radius'],
                        'candidate_index':best,'n_bins':specs[best]['n_bins'],
                        'selection_accuracy':float(mean[best]),
                        'baseline_r_0_1_accuracy':float(mean[3]),'full_accuracy':float(mean[-1]),
                        'n_tied_best':len(tied),'tied_best_candidates':[specs[i]['name'] for i in tied],
                        'tied_candidate_indices':tied,'at_or_below_chance_reference':bool(mean[best]<=1/N_IMAGES)})
    controls=[]
    for key in ['clean']+[f'pure_rep_{rep:02d}' for rep in range(5)]:
        q,tied=read_chunk(out,key,state)
        for ci,spec in enumerate(specs):
            controls.append({'control':'clean' if key=='clean' else 'pure_noise',
                             'repetition':None if key=='clean' else int(key[-2:]),
                             'candidate':spec['name'],'radius':spec['radius'],
                             'accuracy':float(q[ci].mean()),
                             'accuracy_a_to_b':float(q[ci,0].mean()),
                             'accuracy_b_to_a':float(q[ci,1].mean()),
                             'tied_query_fraction':float(tied[ci].mean())})
    return scores,summary,repeated,choices,controls


def plot_results(out,scores,choices,specs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180})
    folder=out/'figures'
    folder.mkdir(exist_ok=True)
    ts=np.asarray(TIMESTEPS)
    numeric=np.array([np.nan if r['radius'] is None else r['radius'] for r in choices])
    fig,(full_ax,ax)=plt.subplots(2,1,figsize=(11,6.7),sharex=True,
                                  gridspec_kw={'height_ratios':[1,6],'hspace':.10})
    for ti,row in enumerate(choices):
        for ci in row['tied_candidate_indices']:
            if ci==row['candidate_index']:
                continue
            r=specs[ci]['radius']
            target=full_ax if r is None else ax
            target.scatter(row['timestep'],.5 if r is None else r,
                           marker='s',s=13,color='#9ca3af',alpha=.55,zorder=1)
    ax.plot(ts,numeric,color='#1864ab',linewidth=1,alpha=.65)
    ax.scatter(ts,numeric,color='#1864ab',s=21,zorder=3,label='Selected R (smallest among tied best)')
    full_ts=[r['timestep'] for r in choices if r['radius'] is None]
    full_ax.scatter(full_ts,[.5]*len(full_ts),color='#1864ab',s=24,zorder=3)
    ax.scatter([],[],color='#9ca3af',marker='s',s=24,label='Other tied-best candidates')
    full_ax.set_ylim(0,1)
    full_ax.set_yticks([.5],['Full spectrum'])
    full_ax.text(.99,.85,'Separate option, not a numeric R',transform=full_ax.transAxes,
                 ha='right',fontsize=9,color='#555555')
    ax.set_ylim(0,.52)
    ax.set_yticks([0,.025,.05,.1,.15,.2,.3,.4,.5])
    ax.set_ylabel('Selected cutoff R (cycles per prepared-image pixel)')
    ax.set_xlabel('Diffusion timestep t')
    ax.set_xlim(0,999)
    ax.set_xticks([0,100,250,500,750,900,999])
    ax.grid(alpha=.2)
    ax.legend(loc='upper right',fontsize=9)
    fig.suptitle(f'Selection-chosen Fourier cutoff versus noise timestep | {N_IMAGES:,} images')
    fig.subplots_adjust(bottom=.17,left=.12,top=.9)
    fig.text(.12,.055,'Smallest-R tie rule; points are measured choices, not a fitted law. Near-chance winners may be unreliable.\n'
             'Selection only; source: selected_cutoffs.csv. The full-spectrum option includes all non-DC bins.',fontsize=9)
    fig.savefig(folder/'selected_R_vs_t.png')
    plt.close(fig)
    # Exact numerical timestep coordinates, including the shorter 990-to-999 interval.
    edges=np.r_[ts[0]-(ts[1]-ts[0])/2,(ts[:-1]+ts[1:])/2,ts[-1]+(ts[-1]-ts[-2])/2]
    fig,ax=plt.subplots(figsize=(11,5.8))
    mesh=ax.pcolormesh(edges,np.arange(13)-.5,scores.mean(axis=0).T*100,
                       shading='flat',vmin=0,vmax=100,cmap='viridis')
    ax.set_yticks(range(12),[s['name'] for s in specs])
    ax.set_xlim(0,999)
    ax.set_xticks([0,100,250,500,750,900,999])
    ax.set_xlabel('Diffusion timestep t')
    ax.set_ylabel('Candidate cutoff (ordered categories)')
    ax.set_title('Selection matching accuracy across cutoffs and timesteps')
    fig.colorbar(mesh,ax=ax,label='Mean matching accuracy (%)')
    fig.subplots_adjust(bottom=.2,left=.12,right=.92)
    fig.text(.12,.055,f'{N_IMAGES:,} candidates/query; five noise pairs; both directions. Chance = {100/N_IMAGES:g}%.\n'
             'Source: accuracy_summary.csv. No evaluation scores are shown.',fontsize=9)
    fig.savefig(folder/'selection_accuracy_heatmap.png')
    plt.close(fig)
    best=np.array([r['candidate_index'] for r in choices])
    chosen=np.stack([scores[:,ti,ci] for ti,ci in enumerate(best)],axis=1)
    fig,ax=plt.subplots(figsize=(11,5.8))
    for values,label,color in [(chosen,'Selection-chosen R(t)','#1864ab'),
                               (scores[:,:,3],'Fixed R = 0.1','#d97706'),
                               (scores[:,:,-1],'Full spectrum','#7c3aed')]:
        ax.plot(ts,values.mean(axis=0)*100,label=label,color=color,lw=1.8)
        ax.fill_between(ts,values.min(axis=0)*100,values.max(axis=0)*100,color=color,alpha=.12)
    ax.axhline(100/N_IMAGES,color='#555555',ls='--',lw=1,label=f'Chance ({100/N_IMAGES:g}%)')
    ax.set(xlim=(0,999),ylim=(0,102),xlabel='Diffusion timestep t',ylabel='Matching accuracy (%)',
           title='In-sample selection accuracy — not held-out performance')
    ax.set_xticks([0,100,250,500,750,900,999])
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.subplots_adjust(bottom=.21,left=.09)
    fig.text(.09,.06,'Shading: minimum–maximum over five noise repetitions, NOT confidence intervals.\n'
             'R(t) was selected using these same images/repetitions. Source: accuracy_by_repeat.csv and selected_cutoffs.csv.',fontsize=9)
    fig.savefig(folder/'selection_chosen_vs_baselines.png')
    plt.close(fig)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('/yunity/poil08/cifar10_cutoff_1000_preparation_001'))
    p.add_argument('--validation-dir',type=Path,default=Path('/yunity/poil08/cifar10_cutoff_1000_validation_001'))
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--check-only',action='store_true',help='Validate inputs and Script 2 provenance; no output or sweep')
    p.add_argument('--resume',action='store_true',help='Continue an incomplete matching run with exactly the same inputs/code/settings')
    args=p.parse_args(argv)
    if min(args.batch_size,args.threads)<1:
        p.error('Batch size and threads must be positive')
    return args


def _run(args):
    dependencies(args.threads)
    root=args.input_dir.resolve()
    out=args.out_dir.resolve()
    validation_root=args.validation_dir.resolve()
    require(not out.is_relative_to(root) and not root.is_relative_to(out)
            and not out.is_relative_to(validation_root) and not validation_root.is_relative_to(out),
            'Output must be separate from preparation and validation')
    require(not out.is_symlink(),'Symlinked output not supported')
    if args.resume:
        require(out.is_dir(),'--resume requires an existing incomplete output')
    else:
        require(not out.exists(),'Output exists; choose a new folder or use --resume for an incomplete run')
    source=Preparation(root)
    validation=load_validation(validation_root/'validation_report.json',source)
    masks,specs=masks_and_specs()
    versions={s:importlib.metadata.version(s) for s in ('torch','numpy','scipy','Pillow','matplotlib')}
    jobs=['clean']+[f'pure_rep_{r:02d}' for r in range(5)]+[
        f'rep_{r:02d}_t_{t:04d}' for r in range(5) for t in TIMESTEPS]
    fingerprint={'input_sha256':source.input_hashes,'validation':validation,
                 'script_sha256':file_hash(__file__),'versions':versions,
                 'batch_size':args.batch_size,'threads':args.threads,
                 'n_images':N_IMAGES,'timesteps':TIMESTEPS,'candidate_specs':specs}
    state=None
    if args.resume:
        require(not (out/'analysis_manifest.json').is_symlink(),'Symlinked manifest not supported')
        state=json.loads((out/'analysis_manifest.json').read_text())
        require(state['fingerprint']==fingerprint,'Resume inputs/code/settings differ; refusing mixed results')
        require(state['status'] in ('running','failed','interrupted'),'Only incomplete runs can resume')
        require(set(state['chunks']).issubset(jobs),'Unexpected completed job in resume state')
        for key in state['chunks']:
            read_chunk(out,key,state)
        print(f'Resume verified: {len(state["chunks"])}/{len(jobs)} completed jobs',flush=True)
    if args.check_only:
        print(f'Preflight passed: {N_IMAGES} selection images; 101 timesteps; 12 cutoffs; 5 A/B repetitions.',flush=True)
        print('Script 2 provenance matches. No output written or cutoff search performed.',flush=True)
        return 0
    if state is None:
        out.mkdir(parents=True,exist_ok=False)
        (out/'chunks').mkdir()
        state={'schema_version':3,'experiment':'selection_only_cutoff_sweep_v3','status':'running',
               'created_utc':datetime.now(timezone.utc).isoformat(),'fingerprint':fingerprint,
               'files_read':source.read_files,'evaluation_tensors_read':False,
               'evaluation_noise_generated':False,'evaluation_shared_metadata_read':True,
               'chunks':{},'outputs':[],'total_jobs':len(jobs)}
    else:
        state['status']='running'
        state.pop('error',None)
    atomic_json(out/'analysis_manifest.json',state)
    start=time.perf_counter()
    try:
        # Recheck numerical agreement in the actual sweep environment.
        state['numerical_checks']=numerical_checks(source,masks,args.batch_size)
        protocol=dict(source.protocol)
        protocol.update({'schema_version':3,'status':'selection_protocol_frozen',
                         'source_preparation_sha256':source.input_hashes['manifest.json'],
                         'selection_sample_ids':[r['sample_id'] for r in source.m['groups']['selection']['records']],
                         'timesteps':TIMESTEPS,'repetitions':5,'candidate_specs':specs,
                         'engine':'CPU float64 FFT and cumulative-annulus RMS with row-batched scipy cdist',
                         'per_query_storage':'chunks/*.npz: credit and tied, axes [candidate, direction, saved sample order]',
                         'direction_order':['A_to_B','B_to_A'],'versions':versions})
        atomic_json(out/'protocol.json',protocol)
        if 'clean' not in state['chunks']:
            features=fft_features(source.clean,args.batch_size)
            q,tied=score_pair(features,features,masks,args.batch_size)
            save_chunk(out,'clean',q,tied,state)
            del features
            print('Saved full selection clean control',flush=True)
        for rep in range(5):
            pending=[t for t in TIMESTEPS if f'rep_{rep:02d}_t_{t:04d}' not in state['chunks']]
            pure_key=f'pure_rep_{rep:02d}'
            if not pending and pure_key in state['chunks']:
                continue
            ea=noise(source.rows[('selection',rep,'A')]['seeds_in_saved_sample_order'])
            eb=noise(source.rows[('selection',rep,'B')]['seeds_in_saved_sample_order'])
            for bank,eps in [('A',ea),('B',eb)]:
                require(tensor_hash(eps)==source.rows[('selection',rep,bank)]['noise_tensor_sha256'],
                        'Noise regeneration mismatch during sweep')
            if pure_key not in state['chunks']:
                q,tied=score_pair(fft_features(ea,args.batch_size),fft_features(eb,args.batch_size),
                                  masks,args.batch_size)
                save_chunk(out,pure_key,q,tied,state)
            for t in pending:
                tick=time.perf_counter()
                a=fft_features(noisy(source.clean,ea,source.alpha,t),args.batch_size)
                b=fft_features(noisy(source.clean,eb,source.alpha,t),args.batch_size)
                q,tied=score_pair(a,b,masks,args.batch_size)
                save_chunk(out,f'rep_{rep:02d}_t_{t:04d}',q,tied,state)
                print(f'Saved selection pair {rep+1}/5, t={t}: {len(state["chunks"])}/{len(jobs)} jobs '
                      f'({time.perf_counter()-tick:.1f}s)',flush=True)
                del a,b
            del ea,eb
        require(set(state['chunks'])==set(jobs),'Missing sweep/control jobs')
        scores,summary,repeated,choices,controls=summarize(out,state,specs)
        for name,digest in source.input_hashes.items():
            require(file_hash(root/name)==digest,f'Input changed during run: {name}')
        require(file_hash(validation['path'])==validation['sha256'],'Validation report changed during sweep')
        save_csv(out/'accuracy_summary.csv',summary)
        save_csv(out/'accuracy_by_repeat.csv',repeated)
        save_csv(out/'control_accuracy.csv',controls)
        csv_choices=[{**r,'tied_best_candidates':';'.join(r['tied_best_candidates']),
                      'tied_candidate_indices':';'.join(map(str,r['tied_candidate_indices']))} for r in choices]
        save_csv(out/'selected_cutoffs.csv',csv_choices)
        atomic_json(out/'selected_cutoffs.json',{
            'schema_version':3,'experiment':'frozen_selection_cutoffs_v3',
            'source_preparation_sha256':source.input_hashes['manifest.json'],
            'protocol_sha256':file_hash(out/'protocol.json'),'n_selection_images':N_IMAGES,
            'choices':choices,'n_timesteps_with_ties':sum(r['n_tied_best']>1 for r in choices),
            'selection_only':True,'evaluation_used':False})
        plot_results(out,scores,choices,specs)
        files=['protocol.json','accuracy_summary.csv','accuracy_by_repeat.csv','control_accuracy.csv',
               'selected_cutoffs.csv','selected_cutoffs.json',
               'figures/selected_R_vs_t.png','figures/selection_accuracy_heatmap.png',
               'figures/selection_chosen_vs_baselines.png']
        state['outputs']=[{'file':name,'sha256':file_hash(out/name)} for name in files]
        atomic_json(out/'freeze_receipt.json',{
            'source_preparation_sha256':source.input_hashes['manifest.json'],
            'protocol_sha256':file_hash(out/'protocol.json'),
            'selected_cutoffs_sha256':file_hash(out/'selected_cutoffs.json'),
            'instruction':'Preserve these files unchanged. Evaluation must use this saved schedule, not rerun cutoff search.'})
        state['outputs'].append({'file':'freeze_receipt.json','sha256':file_hash(out/'freeze_receipt.json')})
        state['status']='complete'
        state['completed_utc']=datetime.now(timezone.utc).isoformat()
        state['last_invocation_seconds']=time.perf_counter()-start
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        state['process_peak_rss_mib']=(peak if sys.platform=='darwin' else peak*1024)/1024**2
        state['cutoff_search_performed']=True
        atomic_json(out/'analysis_manifest.json',state)
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        state['error']=f'{type(error).__name__}: {error}'
        atomic_json(out/'analysis_manifest.json',state)
        raise
    print(f'Complete: {out}',flush=True)
    print('101 selection-derived cutoff choices saved; evaluation NOT opened or scored.',flush=True)
    print(f'Main plot: {out / "figures/selected_R_vs_t.png"}',flush=True)
    print('Keep frozen choices unchanged before held-out evaluation. Near-chance winners are not reliable optima.',flush=True)
    return 0


def run(args):
    if args.check_only:
        return _run(args)
    out=args.out_dir.resolve()
    for source in (args.input_dir.resolve(),args.validation_dir.resolve()):
        require(not out.is_relative_to(source) and not source.is_relative_to(out),
                'Output must be separate from preparation and validation')
    out.parent.mkdir(parents=True,exist_ok=True)
    # A sibling lock also protects directory creation and resume-state reads.
    with (out.parent/f'.{out.name}.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is using this output; do not resume a running job')
        return _run(args)


def main(argv=None):
    return run(parse_args(argv))


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}',file=sys.stderr)
        sys.exit(1)

