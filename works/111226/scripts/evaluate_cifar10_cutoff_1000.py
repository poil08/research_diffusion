#!/usr/bin/env python3
"""Step 4: held-out evaluation of frozen v3 CIFAR-10 cutoffs.
Standalone CPU script. Only evaluation tensors and noise; no cutoff search.
Verifies frozen selection provenance, saves resumable scores and comparison plots.
"""
import argparse
import csv
import os
import shutil
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
    """Only shared metadata and evaluation/clean.pt can be opened."""
    ALLOWED={'manifest.json','noise_plan.json','planned_protocol.json','split.json',
             'selected_images.json','schedule.pt','evaluation/clean.pt'}

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
        payload=self.read_tensor('evaluation/clean.pt')
        self.clean=payload['samples']
        original=payload['original_uint8']
        group=self.m['groups']['evaluation']
        require(payload['group']=='evaluation' and payload['records']==group['records']
                and payload['sample_ids']==splits['evaluation'], 'Evaluation tensor ID/order mismatch')
        require(self.clean.dtype==torch.float32 and tuple(self.clean.shape)==(N_IMAGES,3,64,64)
                and torch.isfinite(self.clean).all() and self.clean.min()>=-1 and self.clean.max()<=1,
                'Invalid clean tensor')
        require(original.dtype==torch.uint8 and tuple(original.shape)==(N_IMAGES,3,32,32),
                'Invalid original tensor')
        require(tensor_hash(self.clean)==group['clean_tensor_sha256']
                and tensor_hash(original)==group['original_tensor_sha256'],'Evaluation tensor hash mismatch')
        seen=set()
        for i in range(N_IMAGES):
            im=Image.fromarray(original[i].permute(1,2,0).numpy())
            arr=np.array(im.resize((64,64),Image.Resampling.BILINEAR),copy=True)
            expected=torch.from_numpy(arr).permute(2,0,1).float().div(127.5).sub(1)
            require(torch.equal(expected,self.clean[i]), 'Resizing/normalization reconstruction mismatch')
            h=tensor_hash(self.clean[i])
            require(h not in seen, 'Duplicate prepared evaluation image')
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
                row=self.rows[('evaluation',rep,bank)]
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
            print(f'Validated evaluation noise pair {rep+1}/5',flush=True)

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



METHODS=['frozen_cutoff','fixed_r_0_1','full_spectrum']


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


class FrozenSelection:
    """Read selection results/provenance, never selection image tensors/chunks."""
    SNAPSHOTS=['analysis_manifest.json','protocol.json','selected_cutoffs.json','freeze_receipt.json']

    def __init__(self,root,source,specs):
        self.root=Path(root).resolve(strict=True)
        self.hashes={}
        self.manifest=self.read_json('analysis_manifest.json')
        m=self.manifest
        require(m.get('schema_version')==3 and m.get('status')=='complete'
                and m.get('experiment')=='selection_only_cutoff_sweep_v3'
                and m.get('evaluation_tensors_read') is False
                and m.get('evaluation_noise_generated') is False
                and m.get('cutoff_search_performed') is True,
                'Expected a completed selection-only v3 sweep')
        expected_outputs={'protocol.json','accuracy_summary.csv','accuracy_by_repeat.csv','control_accuracy.csv',
                          'selected_cutoffs.csv','selected_cutoffs.json','freeze_receipt.json',
                          'figures/selected_R_vs_t.png','figures/selection_accuracy_heatmap.png',
                          'figures/selection_chosen_vs_baselines.png'}
        entries=m['outputs']
        require(len(entries)==len(expected_outputs) and {r['file'] for r in entries}==expected_outputs,
                'Selection output inventory is incomplete or unexpected')
        for row in entries:
            require(self.digest(row['file'])==row['sha256'],f'Selection artifact changed: {row["file"]}')
        fp=m['fingerprint']
        expected_inputs=dict(source.input_hashes)
        expected_inputs.pop('evaluation/clean.pt')
        expected_inputs['selection/clean.pt']=source.artifacts['selection/clean.pt']
        require(fp['input_sha256']==expected_inputs,'Selection belongs to a different preparation')
        require(fp['n_images']==N_IMAGES and fp['timesteps']==TIMESTEPS and fp['candidate_specs']==specs,
                'Selection experiment dimensions/candidate masks differ')
        for name in ('torch','numpy','scipy','Pillow','matplotlib'):
            require(fp['versions'][name]==importlib.metadata.version(name),
                    f'{name} version differs from selection environment')
        jobs={'clean'}|{f'pure_rep_{r:02d}' for r in range(5)}|{
            f'rep_{r:02d}_t_{t:04d}' for r in range(5) for t in TIMESTEPS}
        require(m['total_jobs']==len(jobs) and set(m['chunks'])==jobs,
                'Selection manifest reports missing jobs')
        self.protocol=self.read_json('protocol.json')
        p=self.protocol
        for key,value in source.protocol.items():
            if key!='status':
                require(p.get(key)==value,f'Frozen method changed: {key}')
        require(p['schema_version']==3 and p['status']=='selection_protocol_frozen'
                and p['source_preparation_sha256']==source.input_hashes['manifest.json']
                and p['selection_sample_ids']==[r['sample_id'] for r in source.m['groups']['selection']['records']]
                and p['timesteps']==TIMESTEPS and p['repetitions']==5
                and p['candidate_specs']==specs and p['versions']==fp['versions'],
                'Frozen protocol identity mismatch')
        obj=self.read_json('selected_cutoffs.json')
        require(obj['schema_version']==3 and obj['experiment']=='frozen_selection_cutoffs_v3'
                and obj['source_preparation_sha256']==source.input_hashes['manifest.json']
                and obj['protocol_sha256']==self.hashes['protocol.json']
                and obj['n_selection_images']==N_IMAGES
                and obj['selection_only'] is True and obj['evaluation_used'] is False,
                'Invalid frozen cutoff provenance')
        self.choices=obj['choices']
        require([r['timestep'] for r in self.choices]==TIMESTEPS,'Frozen timestep schedule is incomplete or reordered')
        for row in self.choices:
            ci=row['candidate_index']
            require(type(ci) is int and 0<=ci<len(specs),'Invalid frozen candidate index')
            spec=specs[ci]
            require(row['radius']==spec['radius'] and row['candidate']==spec['name']
                    and row['n_bins']==spec['n_bins'],'Frozen radius/name/index mismatch')
            tied=row['tied_candidate_indices']
            require(tied==sorted(set(tied)) and tied and ci==tied[0]
                    and all(type(i) is int and 0<=i<len(specs) for i in tied)
                    and row['n_tied_best']==len(tied)
                    and row['tied_best_candidates']==[specs[i]['name'] for i in tied],
                    'Frozen tie record is inconsistent')
            for key in ('selection_accuracy','baseline_r_0_1_accuracy','full_accuracy'):
                require(type(row[key]) in (int,float) and np.isfinite(row[key]) and 0<=row[key]<=1,
                        'Invalid recorded selection accuracy')
        require(obj['n_timesteps_with_ties']==sum(r['n_tied_best']>1 for r in self.choices),
                'Frozen tie summary mismatch')
        receipt=self.read_json('freeze_receipt.json')
        require(receipt['source_preparation_sha256']==source.input_hashes['manifest.json']
                and receipt['protocol_sha256']==self.hashes['protocol.json']
                and receipt['selected_cutoffs_sha256']==self.hashes['selected_cutoffs.json'],
                'Freeze receipt mismatch')
        self.indices={r['timestep']:[r['candidate_index'],3,11] for r in self.choices}
        self.control_indices=sorted({i for values in self.indices.values() for i in values})

    def path(self,name):
        require(name=='analysis_manifest.json' or name in {
            'protocol.json','accuracy_summary.csv','accuracy_by_repeat.csv','control_accuracy.csv',
            'selected_cutoffs.csv','selected_cutoffs.json','freeze_receipt.json',
            'figures/selected_R_vs_t.png','figures/selection_accuracy_heatmap.png',
            'figures/selection_chosen_vs_baselines.png'},'Non-allowlisted selection file')
        path=self.root/name
        require(path.is_file() and path.resolve().is_relative_to(self.root)
                and not any(p.is_symlink() for p in [path,*path.parents]),'Unsafe/missing selection file')
        return path

    def digest(self,name):
        digest=file_hash(self.path(name))
        self.hashes[name]=digest
        return digest

    def read_json(self,name):
        self.digest(name)
        return json.loads(self.path(name).read_text())

    def snapshot(self,out):
        folder=out/'frozen_inputs'
        folder.mkdir(exist_ok=True)
        for name in self.SNAPSHOTS:
            target=folder/name
            if target.exists():
                require(not target.is_symlink() and file_hash(target)==self.hashes[name],
                        'Frozen snapshot changed; refusing resume')
            else:
                shutil.copyfile(self.path(name),target)
            require(file_hash(target)==self.hashes[name],'Frozen copy hash mismatch')


def score_indices(a,b,masks,indices,batch_size):
    """Score only requested candidates; duplicate methods share identical results.

    Accumulate the original nested annuli so floating-point reduction order matches
    Script 3. Full spectrum requires all bins, but no other retrieval scores or
    winners are computed.
    """
    found={}
    for ci,d in enumerate(distance_sweep(a,b,masks,batch_size)):
        if ci in indices:
            ab,na=credits(d)
            ba,nb=credits(d.T)
            found[ci]=(np.stack([ab,ba]),np.stack([na>1,nb>1]))
    return (np.stack([found[i][0] for i in indices]),
            np.stack([found[i][1] for i in indices]))


def numerical_checks(source,masks,frozen,batch_size):
    """Small technical checks, with already-frozen method indices only."""
    n=min(16,len(source.clean))
    clean=source.clean[:n]
    ea=noise(source.rows[('evaluation',0,'A')]['seeds_in_saved_sample_order'][:n])
    eb=noise(source.rows[('evaluation',0,'B')]['seeds_in_saved_sample_order'][:n])
    errors=[]
    for t in (0,500,999):
        require(t in frozen.indices,'Missing required probe timestep')
        x=noisy(clean,ea,source.alpha,t)
        y=noisy(clean,eb,source.alpha,t)
        a,b=fft_features(x,batch_size),fft_features(y,batch_size)
        qa,ta=score_indices(a,b,masks,frozen.indices[t],batch_size)
        qb,tb=score_indices(fft_features(x,1),fft_features(y,1),masks,frozen.indices[t],1)
        require(np.array_equal(qa,qb) and np.array_equal(ta,tb),'Batch size changes retrieval/ties')
        for ci,d in enumerate(distance_sweep(a,b,masks,batch_size)):
            if ci not in frozen.indices[t]:
                continue
            mask=masks[ci]
            direct=np.sqrt(np.mean((a[:,mask][:,None,:]-b[:,mask][None,:,:])**2,axis=-1))
            require(np.allclose(d,direct,atol=1e-12,rtol=1e-12),'Batched/direct distances disagree')
            for mat,ref in ((d,direct),(d.T,direct.T)):
                require(np.array_equal(credits(mat)[0],credits(ref)[0]),'Direct retrieval differs')
            errors.append(float(np.max(np.abs(d-direct))))
    return {'probe_images':n,'timesteps':[0,500,999],'max_distance_error':max(errors),
            'batch_and_direct_credit_agreement':True,'only_frozen_methods_scored':True}


def save_chunk(out,key,q,tied,indices,state):
    path=out/'chunks'/(key+'.npz')
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,credit=q,tied=tied,indices=np.array(indices,dtype=np.int64),key=np.array(key))
    os.replace(tmp,path)
    state['chunks'][key]={'file':str(path.relative_to(out)),'sha256':file_hash(path)}
    state['updated_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(out/'analysis_manifest.json',state)


def read_chunk(out,key,indices,state):
    out=Path(out).resolve()
    entry=state['chunks'][key]
    require(entry['file']==f'chunks/{key}.npz','Unexpected chunk path')
    path=out/entry['file']
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(out)
            and file_hash(path)==entry['sha256'],'Evaluation chunk missing or changed')
    with np.load(path,allow_pickle=False) as z:
        require(str(z['key'])==key and z['indices'].tolist()==indices,'Chunk method order mismatch')
        q,tied=z['credit'].copy(),z['tied'].copy()
    require(q.dtype==np.float64 and q.shape==(len(indices),2,N_IMAGES)
            and tied.dtype==np.bool_ and tied.shape==q.shape
            and np.isfinite(q).all() and ((q>=0)&(q<=1)).all(),'Invalid evaluation credits')
    return q,tied


def summarize(out,state,frozen,specs):
    scores=np.empty((5,len(TIMESTEPS),3))
    repeated,summary,paired_repeated,paired,cohorts=[],[],[],[],[]
    for rep in range(5):
        for ti,t in enumerate(TIMESTEPS):
            ix=frozen.indices[t]
            q,tied=read_chunk(out,f'rep_{rep:02d}_t_{t:04d}',ix,state)
            scores[rep,ti]=q.mean(axis=(1,2))
            for mi,method in enumerate(METHODS):
                repeated.append({'repetition':rep,'timestep':t,'method':method,
                                 'candidate':specs[ix[mi]]['name'],'radius':specs[ix[mi]]['radius'],
                                 'accuracy':float(scores[rep,ti,mi]),
                                 'accuracy_a_to_b':float(q[mi,0].mean()),
                                 'accuracy_b_to_a':float(q[mi,1].mean()),
                                 'tied_query_fraction':float(tied[mi].mean())})
            for bi in (1,2):
                paired_repeated.append({'repetition':rep,'timestep':t,'baseline':METHODS[bi],
                                        'frozen_minus_baseline_pp':float((scores[rep,ti,0]-scores[rep,ti,bi])*100)})
    for ti,t in enumerate(TIMESTEPS):
        for mi,method in enumerate(METHODS):
            values=scores[:,ti,mi]
            ci=frozen.indices[t][mi]
            summary.append({'timestep':t,'method':method,'candidate':specs[ci]['name'],
                            'radius':specs[ci]['radius'],'mean_accuracy':float(values.mean()),
                            'min_repeat_accuracy':float(values.min()),'max_repeat_accuracy':float(values.max())})
        for bi in (1,2):
            values=(scores[:,ti,0]-scores[:,ti,bi])*100
            paired.append({'timestep':t,'baseline':METHODS[bi],'mean_difference_pp':float(values.mean()),
                           'min_repeat_difference_pp':float(values.min()),'max_repeat_difference_pp':float(values.max())})
        selected=frozen.choices[ti]['selection_accuracy']
        evaluated=float(scores[:,ti,0].mean())
        cohorts.append({'timestep':t,'radius':frozen.choices[ti]['radius'],
                        'selection_accuracy':selected,'evaluation_accuracy':evaluated,
                        'evaluation_minus_selection_pp':(evaluated-selected)*100})
    controls=[]
    for key in ['clean']+[f'pure_rep_{rep:02d}' for rep in range(5)]:
        q,tied=read_chunk(out,key,frozen.control_indices,state)
        for j,ci in enumerate(frozen.control_indices):
            controls.append({'control':'clean' if key=='clean' else 'pure_noise',
                             'repetition':None if key=='clean' else int(key[-2:]),
                             'candidate':specs[ci]['name'],'radius':specs[ci]['radius'],
                             'accuracy':float(q[j].mean()),'accuracy_a_to_b':float(q[j,0].mean()),
                             'accuracy_b_to_a':float(q[j,1].mean()),'tied_query_fraction':float(tied[j].mean())})
    return scores,summary,repeated,paired,paired_repeated,cohorts,controls


def plot_results(out,scores,frozen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    folder=out/'figures'
    folder.mkdir(exist_ok=True)
    ts=np.array(TIMESTEPS)
    colors=['#1864ab','#d97706','#7c3aed']
    labels=['Frozen selection-chosen R(t)','Fixed R = 0.1','Full spectrum']
    def finish(fig,ax,name,note):
        ax.set_xlim(0,999)
        ax.set_xticks([0,100,250,500,750,900,999])
        ax.set_xlabel('Diffusion timestep t')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        fig.subplots_adjust(left=.10,right=.96,bottom=.24,top=.9)
        fig.text(.10,.06,note,fontsize=9)
        fig.savefig(folder/name)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    for mi,label in enumerate(labels):
        x=scores[:,:,mi]*100
        ax.plot(ts,x.mean(axis=0),color=colors[mi],label=label,lw=1.8)
        ax.fill_between(ts,x.min(axis=0),x.max(axis=0),color=colors[mi],alpha=.12)
    ax.axhline(100/N_IMAGES,ls='--',color='#555555',label=f'Chance ({100/N_IMAGES:g}%)')
    ax.set_ylim(0,102)
    ax.set_ylabel('Matching accuracy (%)')
    ax.set_title(f'Held-out matching accuracy | {N_IMAGES:,} evaluation images')
    finish(fig,ax,'evaluation_accuracy.png',
           'Frozen choices applied unchanged; no evaluation cutoff search. Shading: repeat minimum–maximum, NOT confidence intervals.\n'
           'Five noise pairs reuse the same original images. Source: accuracy_by_repeat.csv.')
    fig,ax=plt.subplots(figsize=(11,6))
    for mi in (1,2):
        diff=(scores[:,:,0]-scores[:,:,mi])*100
        ax.plot(ts,diff.mean(axis=0),color=colors[mi],label='Frozen minus '+labels[mi],lw=1.8)
        ax.fill_between(ts,diff.min(axis=0),diff.max(axis=0),color=colors[mi],alpha=.12)
    ax.axhline(0,color='#555555',ls='--')
    ax.set_ylabel('Accuracy difference (percentage points)')
    ax.set_title('Held-out advantage over baselines')
    finish(fig,ax,'advantage_over_baselines.png',
           'Positive values favor frozen choices; negative values favor the baseline. Differences use the same evaluation images/noise.\n'
           'Shading: paired repeat minimum–maximum, NOT confidence intervals. Source: paired_differences_by_repeat.csv.')
    fig,ax=plt.subplots(figsize=(11,6))
    ax.plot(ts,[r['selection_accuracy']*100 for r in frozen.choices],label='Selection (in-sample)',color='#6b7280',lw=1.8)
    ax.plot(ts,scores[:,:,0].mean(axis=0)*100,label='Evaluation (held-out)',color=colors[0],lw=1.8)
    ax.axhline(100/N_IMAGES,color='#555555',ls='--',label=f'Chance ({100/N_IMAGES:g}%)')
    ax.set_ylim(0,102)
    ax.set_ylabel('Matching accuracy (%)')
    ax.set_title('Same frozen cutoff schedule, different original images')
    finish(fig,ax,'selection_vs_evaluation.png',
           'Groups use different original images and noise. The gap cannot be attributed solely to selection optimism.\n'
           'Source: selection_vs_evaluation.csv and frozen_inputs/selected_cutoffs.json.')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('/yunity/poil08/cifar10_cutoff_1000_preparation_001'))
    p.add_argument('--selection-dir',type=Path,default=Path('/yunity/poil08/cifar10_cutoff_1000_selection_001'))
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--check-only',action='store_true',help='Verify frozen files and evaluation input/noise hashes; no metrics or outputs')
    p.add_argument('--resume',action='store_true',help='Resume an incomplete run with identical code, inputs and settings')
    args=p.parse_args(argv)
    if min(args.batch_size,args.threads)<1:
        p.error('Batch size and threads must be positive')
    return args


def _run(args):
    dependencies(args.threads)
    root,out,selection=args.input_dir.resolve(),args.out_dir.resolve(),args.selection_dir.resolve()
    for src in (root,selection):
        require(not out.is_relative_to(src) and not src.is_relative_to(out),'Output must be separate from inputs')
    if args.resume:
        require(out.is_dir(),'Resume requires an existing incomplete output')
    else:
        require(not out.exists(),'Output exists; choose a new folder or resume an incomplete run')
    source=Preparation(root)
    masks,specs=masks_and_specs()
    frozen=FrozenSelection(selection,source,specs)
    print(f'Frozen schedule verified: {len(frozen.choices)} choices; no retuning permitted.',flush=True)
    versions={s:importlib.metadata.version(s) for s in ('torch','numpy','scipy','Pillow','matplotlib')}
    jobs={'clean':frozen.control_indices}
    jobs.update({f'pure_rep_{r:02d}':frozen.control_indices for r in range(5)})
    jobs.update({f'rep_{r:02d}_t_{t:04d}':frozen.indices[t] for r in range(5) for t in TIMESTEPS})
    fingerprint={'preparation_input_sha256':source.input_hashes,'selection_input_sha256':frozen.hashes,
                 'script_sha256':file_hash(__file__),'versions':versions,
                 'batch_size':args.batch_size,'threads':args.threads,
                 'n_images':N_IMAGES,'timesteps':TIMESTEPS,'applied_indices':[
                     frozen.indices[t] for t in TIMESTEPS]}
    state=None
    if args.resume:
        state=json.loads((out/'analysis_manifest.json').read_text())
        require(state['fingerprint']==fingerprint,'Resume inputs/code/settings differ')
        require(state['status'] in ('running','failed','interrupted'),'Only incomplete runs can resume')
        require(set(state['chunks']).issubset(jobs),'Unexpected saved evaluation job')
        for key in state['chunks']:
            read_chunk(out,key,jobs[key],state)
        for name in frozen.SNAPSHOTS:
            path=out/'frozen_inputs'/name
            require(path.is_file() and not path.is_symlink() and file_hash(path)==frozen.hashes[name],
                    'Frozen snapshot changed or missing')
        print(f'Resume verified: {len(state["chunks"])}/{len(jobs)} completed jobs',flush=True)
    if args.check_only:
        print('Preflight passed: evaluation input/noise hashes and frozen selection provenance verified.',flush=True)
        print('No FFT, metrics, cutoff selection or output writes. Selection image tensors were NOT opened.',flush=True)
        return 0
    if state is None:
        out.mkdir(parents=True,exist_ok=False)
        (out/'chunks').mkdir()
        state={'schema_version':3,'experiment':'heldout_frozen_cutoff_evaluation_v3','status':'running',
               'created_utc':datetime.now(timezone.utc).isoformat(),'fingerprint':fingerprint,
               'preparation_files_read':source.read_files,'selection_results_read':list(frozen.hashes),
               'selection_tensors_read':False,'selection_noise_generated':False,'cutoff_search_performed':False,
               'chunks':{},'outputs':[],'total_jobs':len(jobs),
               'selection_check_scope':'All 10 reported output file hashes and completion/job inventory verified. Selection query chunks not reread/recomputed.'}
    else:
        state['status']='running'
        state.pop('error',None)
    atomic_json(out/'analysis_manifest.json',state)
    start=time.perf_counter()
    try:
        frozen.snapshot(out)
        state['numerical_checks']=numerical_checks(source,masks,frozen,args.batch_size)
        protocol={'schema_version':3,'experiment':'frozen_cutoff_evaluation_v3','group':'evaluation',
                  'n_images':N_IMAGES,'sample_ids':[r['sample_id'] for r in source.m['groups']['evaluation']['records']],
                  'timesteps':TIMESTEPS,'repetitions':5,'method_order':METHODS,
                  'chance_reference':1/N_IMAGES,'cutoff_search_performed':False,
                  'frozen_input_sha256':{k:frozen.hashes[k] for k in frozen.SNAPSHOTS},
                  'preparation_sha256':source.input_hashes['manifest.json'],'versions':versions,
                  'method_definition':'Unchanged copied frozen_inputs/protocol.json',
                  'controls':'Clean and independent pure-noise controls for the union of applied masks only; reused controls across t are not new trials',
                  'chunk_axes':['method_or_control_candidate','direction','evaluation_sample_order'],
                  'direction_order':['A_to_B','B_to_A'],'control_candidate_indices':frozen.control_indices,
                  'paired_differences':'Frozen minus baseline, same evaluation images/noise, percentage points',
                  'limitations':['Five noise repetitions reuse originals; timesteps share epsilon.',
                                 'Repeat min-max is not a confidence interval; no significance test.',
                                 'Do not retune on these results; revised choices need fresh confirmatory data.']}
        atomic_json(out/'evaluation_protocol.json',protocol)
        if 'clean' not in state['chunks']:
            features=fft_features(source.clean,args.batch_size)
            q,tied=score_indices(features,features,masks,frozen.control_indices,args.batch_size)
            save_chunk(out,'clean',q,tied,frozen.control_indices,state)
            del features
        for rep in range(5):
            pending=[t for t in TIMESTEPS if f'rep_{rep:02d}_t_{t:04d}' not in state['chunks']]
            pure_key=f'pure_rep_{rep:02d}'
            if not pending and pure_key in state['chunks']:
                continue
            ea=noise(source.rows[('evaluation',rep,'A')]['seeds_in_saved_sample_order'])
            eb=noise(source.rows[('evaluation',rep,'B')]['seeds_in_saved_sample_order'])
            for bank,eps in [('A',ea),('B',eb)]:
                require(tensor_hash(eps)==source.rows[('evaluation',rep,bank)]['noise_tensor_sha256'],
                        'Noise changed during evaluation')
            if pure_key not in state['chunks']:
                q,tied=score_indices(fft_features(ea,args.batch_size),fft_features(eb,args.batch_size),
                                     masks,frozen.control_indices,args.batch_size)
                save_chunk(out,pure_key,q,tied,frozen.control_indices,state)
            for t in pending:
                tick=time.perf_counter()
                a=fft_features(noisy(source.clean,ea,source.alpha,t),args.batch_size)
                b=fft_features(noisy(source.clean,eb,source.alpha,t),args.batch_size)
                q,tied=score_indices(a,b,masks,frozen.indices[t],args.batch_size)
                save_chunk(out,f'rep_{rep:02d}_t_{t:04d}',q,tied,frozen.indices[t],state)
                print(f'Saved evaluation pair {rep+1}/5, t={t}: {len(state["chunks"])}/{len(jobs)} jobs '
                      f'({time.perf_counter()-tick:.1f}s)',flush=True)
                del a,b
            del ea,eb
        require(set(state['chunks'])==set(jobs),'Missing evaluation jobs')
        scores,summary,repeated,paired,paired_repeated,cohorts,controls=summarize(out,state,frozen,specs)
        for name,digest in source.input_hashes.items():
            require(file_hash(root/name)==digest,f'Preparation input changed: {name}')
        for name,digest in frozen.hashes.items():
            require(file_hash(selection/name)==digest,f'Selection result changed: {name}')
        frozen.snapshot(out)
        save_csv(out/'accuracy_summary.csv',summary)
        save_csv(out/'accuracy_by_repeat.csv',repeated)
        save_csv(out/'paired_differences.csv',paired)
        save_csv(out/'paired_differences_by_repeat.csv',paired_repeated)
        save_csv(out/'selection_vs_evaluation.csv',cohorts)
        save_csv(out/'control_accuracy.csv',controls)
        save_csv(out/'applied_cutoffs.csv',[{
            'timestep':r['timestep'],'candidate':r['candidate'],'radius':r['radius'],
            'candidate_index':r['candidate_index'],'n_bins':r['n_bins']} for r in frozen.choices])
        atomic_json(out/'accuracy_summary.json',summary)
        plot_results(out,scores,frozen)
        files=['evaluation_protocol.json','accuracy_summary.csv','accuracy_summary.json','accuracy_by_repeat.csv',
               'paired_differences.csv','paired_differences_by_repeat.csv','selection_vs_evaluation.csv',
               'control_accuracy.csv','applied_cutoffs.csv',
               'figures/evaluation_accuracy.png','figures/advantage_over_baselines.png','figures/selection_vs_evaluation.png']
        files += ['frozen_inputs/'+name for name in frozen.SNAPSHOTS]
        state['outputs']=[{'file':name,'sha256':file_hash(out/name)} for name in files]
        state['status']='complete'
        state['completed_utc']=datetime.now(timezone.utc).isoformat()
        state['last_invocation_seconds']=time.perf_counter()-start
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        state['process_peak_rss_mib']=(peak if sys.platform=='darwin' else peak*1024)/1024**2
        atomic_json(out/'analysis_manifest.json',state)
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        state['error']=f'{type(error).__name__}: {error}'
        atomic_json(out/'analysis_manifest.json',state)
        raise
    print(f'Complete: {out}',flush=True)
    print(f'{len(TIMESTEPS)} timesteps evaluated using frozen choices and two baselines. NO new cutoffs selected.',flush=True)
    print(f'Main plot: {out / "figures/evaluation_accuracy.png"}',flush=True)
    return 0


def run(args):
    if args.check_only:
        return _run(args)
    out=args.out_dir.resolve()
    for root in (args.input_dir.resolve(),args.selection_dir.resolve()):
        require(not out.is_relative_to(root) and not root.is_relative_to(out),'Output must be separate from inputs')
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/f'.{out.name}.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is using this evaluation output')
        return _run(args)


def main(argv=None):
    return run(parse_args(argv))


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}',file=sys.stderr)
        sys.exit(1)

