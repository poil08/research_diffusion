#!/usr/bin/env python3
"""Step 2: validate v3 preparation and time selection-only FFT matching.

Standalone CPU script. No downloads, cutoff search, evaluation image reads or
changes to preparation. Output must be new. Old v2 preparation is not supported.
"""
import argparse
import csv
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


def benchmark(source,masks,args):
    n=args.benchmark_images
    clean=source.clean[:n]
    # Fixed t=500 and repetition 0 are for timing, not method selection.
    rows=[]
    for iteration in range(args.timing_repeats):
        t0=time.perf_counter()
        ea=noise(source.rows[('selection',0,'A')]['seeds_in_saved_sample_order'][:n])
        eb=noise(source.rows[('selection',0,'B')]['seeds_in_saved_sample_order'][:n])
        t1=time.perf_counter()
        a=fft_features(noisy(clean,ea,source.alpha,500),args.batch_size)
        b=fft_features(noisy(clean,eb,source.alpha,500),args.batch_size)
        t2=time.perf_counter()
        for d in distance_sweep(a,b,masks,args.batch_size):
            credits(d)
            credits(d.T)
        t3=time.perf_counter()
        row={'iteration':iteration+1,'n_images_per_bank':n,'timestep':500,'n_cutoffs':len(masks),
             'noise_seconds':t1-t0,'state_and_fft_seconds':t2-t1,
             'distance_and_retrieval_seconds':t3-t2,'total_seconds':t3-t0}
        rows.append(row)
        print(f'Timing {iteration+1}/{args.timing_repeats}: {n} x {n}, all 12 cutoffs: {t3-t0:.2f} s',flush=True)
    # Linear feature generation, quadratic matching: rough only, not a runtime guarantee.
    factor=N_IMAGES/n
    per_t=[(r['noise_seconds']+r['state_and_fft_seconds'])*factor+
           r['distance_and_retrieval_seconds']*factor**2 for r in rows]
    estimate=[s*len(TIMESTEPS)*5 for s in per_t]
    return rows,{'selection_sweep_hours_range':[min(estimate)/3600,max(estimate)/3600],
                 'basis':'101 timesteps x 5 noise pairs; all 12 cutoffs. Linear FFT/noise and quadratic distance scaling.',
                 'caveat':'Rough estimate, not a confidence interval or guarantee. Excludes startup, controls, plots and output I/O; caching and CPU load can change runtime.'}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('/yunity/poil08/cifar10_cutoff_1000_preparation_001'))
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=64,help='Distance row block and FFT image batch size')
    p.add_argument('--benchmark-images',type=int,default=256,help='First N selection images; 1000 gives full-size timing')
    p.add_argument('--timing-repeats',type=int,default=2,help='Repeated timings of the same inputs, not new noise repetitions')
    p.add_argument('--threads',type=int,default=1,help='Torch threads; scipy cdist here is CPU row-batched')
    p.add_argument('--check-only',action='store_true',help='Verify selection inputs/noise only; no FFT or outputs')
    a=p.parse_args(argv)
    if not 2<=a.benchmark_images<=N_IMAGES or min(a.batch_size,a.timing_repeats,a.threads)<1:
        p.error('Use 2..1000 benchmark images and positive batch size/repeats/threads')
    return a


def run(args):
    dependencies(args.threads)
    root=args.input_dir.resolve()
    out=args.out_dir.resolve()
    require(not out.exists(),'Output already exists; choose a new directory')
    require(not out.is_relative_to(root) and not root.is_relative_to(out),
            'Output must be separate from preparation')
    source=Preparation(root)
    print('Input validation passed: selection tensors and all 10 selection noise banks. Evaluation tensors NOT opened.',flush=True)
    if args.check_only:
        print('Check-only complete: no FFT, timing or output files written.',flush=True)
        return 0
    masks,specs=masks_and_specs()
    checks=numerical_checks(source,masks,args.batch_size)
    print('Numerical checks passed: FFT, direct/batched distances, bidirectional retrieval and repeatability.',flush=True)
    rows,estimate=benchmark(source,masks,args)
    # Verify sources did not change while computations were running.
    for name,digest in source.input_hashes.items():
        require(file_hash(root/name)==digest,f'Input changed during run: {name}')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes=peak if sys.platform=='darwin' else peak*1024
    report={'status':'complete','experiment':'selection_only_v3_validation_and_timing',
            'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':file_hash(__file__),
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'input_sha256':source.input_hashes,'files_read':source.read_files,
            'evaluation_tensors_read':False,'evaluation_noise_generated':False,
            'evaluation_shared_metadata_read':True,'cutoff_search_performed':False,
            'noise_checks':source.noise_checks,'numerical_checks':checks,'candidates':specs,
            'timings':rows,'runtime_estimate':estimate,'process_peak_rss_mib':peak_bytes/1024**2,
            'peak_rss_scope':'Observed entire validation process, not a guarantee for the future full sweep',
            'planned_full_pair_matrix_mib':N_IMAGES**2*8/1024**2,
            'versions':{s:importlib.metadata.version(s) for s in ('torch','numpy','scipy','Pillow')},
            'python_version':platform.python_version(),
            'next_step':'Review validation and timing before running selection. Do not interpret benchmark as final accuracy or tune using evaluation.'}
    out.mkdir(parents=True,exist_ok=False)
    try:
        with (out/'timing.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report['outputs']=[{'file':'timing.csv','sha256':file_hash(out/'timing.csv')}]
        write_json(out/'validation_report.json',report)
    except BaseException as error:
        write_json(out/'FAILED.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
    lo,hi=estimate['selection_sweep_hours_range']
    print(f'Complete: {out}\nObserved peak process memory: {peak_bytes/1024**2:.1f} MiB',flush=True)
    print(f'Rough full selection-sweep estimate: {lo:.2f}–{hi:.2f} hours (not a guarantee).',flush=True)
    print('No cutoff choices were made. No evaluation images were opened or scored.',flush=True)
    return 0


def main(argv=None):
    return run(parse_args(argv))


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}',file=sys.stderr)
        sys.exit(1)
