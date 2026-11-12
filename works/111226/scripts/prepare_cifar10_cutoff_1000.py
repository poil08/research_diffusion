#!/usr/bin/env python3
"""Prepare a 1000/1000 CIFAR-10 cutoff study with lazy, reproducible noise.
Standalone CPU preparation only: no FFT, retrieval, model inference or downloads.
New v3 format: clean images and per-image noise seeds; NOT legacy-analysis compatible.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

MODEL_BASE = Path('/yunity/poil08/hugging_face')
MODEL_FOLDERS = {0: 'hf_cifar10_ddpm_rgb_seed0',
                 42: 'hf_cifar10_ddpm_safe_seed42',
                 123: 'hf_cifar10_ddpm_rgb_seed123'}
SCHEDULE_KEYS = ('num_train_timesteps', 'beta_schedule', 'beta_start', 'beta_end',
                 'trained_betas', 'rescale_betas_zero_snr', 'prediction_type')
GROUPS = ('selection', 'evaluation')



def dependencies():
    global torch, np, Image, ImageDraw, DDPMScheduler
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from diffusers import DDPMScheduler
    torch.set_num_threads(1)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str((str(a.dtype), a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def check_paths(args):
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}. Choose a new --out-dir; nothing overwritten.')
    for source in [args.data_root] + [getattr(args, f'seed{s}_root') for s in MODEL_FOLDERS]:
        if out.is_relative_to(source.resolve()):
            raise ValueError('Output must be outside the dataset and model directories')


def load_schedules(args):
    records, schedulers = {}, []
    for seed in MODEL_FOLDERS:
        path = getattr(args, f'seed{seed}_root') / 'scheduler' / 'scheduler_config.json'
        raw = path.read_bytes()
        config = json.loads(raw)
        missing = set(SCHEDULE_KEYS) - config.keys()
        if missing:
            raise ValueError(f'Seed {seed}: missing schedule keys {sorted(missing)}; verify explicitly')
        if config.get('_class_name') != 'DDPMScheduler':
            raise ValueError(f'Seed {seed}: expected DDPMScheduler configuration')
        sched = DDPMScheduler.from_config(config)
        schedulers.append(sched)
        records[str(seed)] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                              'config': config, 'selected_settings': {k: config[k] for k in SCHEDULE_KEYS},
                              'betas_sha256': tensor_hash(sched.betas)}
    first = records['0']['selected_settings']
    if any(r['selected_settings'] != first for r in records.values()):
        raise ValueError('Selected scheduler settings differ across models')
    if any(not torch.equal(schedulers[0].betas, s.betas) for s in schedulers[1:]):
        raise ValueError('Actual beta tensors differ across models')
    if args.timesteps[-1] >= len(schedulers[0].betas):
        raise ValueError('Requested timestep is outside the saved schedule')
    ab = schedulers[0].alphas_cumprod
    if not torch.isfinite(ab).all() or not ((ab >= 0) & (ab < 1)).all():
        raise ValueError('Invalid cumulative alpha schedule')
    return schedulers[0], records


def load_dataset(args):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=str(args.data_root), train=args.split == 'train', download=False)
    expected = 50000 if args.split == 'train' else 10000
    if len(ds) != expected:
        raise ValueError(f'Expected {expected} images in official {args.split} split')
    files = ds.train_list if args.split == 'train' else ds.test_list
    provenance = [{'path': str((Path(ds.root) / ds.base_folder / name).resolve()),
                   'sha256': file_hash(Path(ds.root) / ds.base_folder / name)} for name, _ in files]
    return ds, provenance


def select_indices(targets, count, seed):
    """Approximately class-balanced, fixed selection. Remainder classes randomized."""
    labels = np.asarray(targets)
    classes = np.unique(labels)
    if count > len(labels):
        raise ValueError('Requested more images than the split contains')
    rng = np.random.default_rng(seed)
    order = rng.permutation(classes)
    q, remainder = divmod(count, len(classes))
    chosen = []
    for rank, label in enumerate(order):
        candidates = np.flatnonzero(labels == label)
        n = q + int(rank < remainder)
        if n > len(candidates):
            raise ValueError('Not enough images to satisfy balanced selection')
        chosen.extend(rng.choice(candidates, n, replace=False).tolist())
    return rng.permutation(chosen).tolist()


def prepare_clean(ds, indices, size, split):
    originals, prepared, records = [], [], []
    for idx in indices:
        im, label = ds[idx]
        if im.mode != 'RGB' or im.size != (32, 32):
            raise ValueError('Expected original RGB 32x32 CIFAR-10 images')
        originals.append(np.array(im, dtype=np.uint8, copy=True))
        enlarged = im.resize((size, size), resample=Image.Resampling.BILINEAR) if size != 32 else im
        prepared.append(np.array(enlarged, dtype=np.uint8, copy=True))
        records.append({'sample_id': f'{split}_{idx:05d}', 'dataset_index': idx,
                        'label': int(label), 'class_name': ds.classes[label]})
    original = torch.from_numpy(np.stack(originals)).permute(0, 3, 1, 2).contiguous()
    rgb = torch.from_numpy(np.stack(prepared)).permute(0, 3, 1, 2).contiguous()
    clean = rgb.float().div(127.5).sub(1).contiguous()
    return original, clean, records


def make_noise(shape, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(shape, generator=generator, device='cpu', dtype=torch.float32)


def add_noise(scheduler, clean, noise, timestep):
    ts = torch.full((len(clean),), timestep, dtype=torch.long)
    result = scheduler.add_noise(clean, noise, ts).contiguous()
    alpha = scheduler.alphas_cumprod[timestep]
    reference = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
    if not torch.isfinite(result).all() or not torch.allclose(result, reference, atol=1e-6, rtol=1e-6):
        raise RuntimeError('Forward-noise equation verification failed')
    return result


def preview_image(tensor):
    # Visualization ONLY: fixed [-1,1] display range, no per-image normalization.
    rgb = ((tensor.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(rgb)


def grid(samples, records, title, destination):
    n = min(16, len(samples))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 156, 48 + rows * 164), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill='black')
    draw.text((8, 25), 'Preview only: values clipped to [-1,1]', fill='black')
    for i in range(n):
        x, y = (i % cols) * 156 + 8, 48 + (i // cols) * 164
        canvas.paste(preview_image(samples[i]).resize((128, 128), Image.Resampling.NEAREST), (x, y))
        draw.text((x, y + 130), records[i]['sample_id'], fill='black')
        draw.text((x, y + 144), records[i]['class_name'], fill='black')
    canvas.save(destination)


def stats(samples):
    return {'min': samples.min().item(), 'max': samples.max().item(),
            'mean': samples.mean().item(), 'std': samples.std(unbiased=False).item(),
            'preview_clipped_fraction': ((samples < -1) | (samples > 1)).float().mean().item()}



N_PER_GROUP = 1000
REPETITIONS = 5
TIMESTEPS = list(range(0, 1000, 10)) + [999]
RADII = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, None]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/yunity/poil08/datasets'))
    p.add_argument('--previous-preparation-dir', type=Path,
                   default=Path('/yunity/poil08/cifar10_cutoff_preparation_001'))
    p.add_argument('--out-dir', type=Path, required=True, help='New directory only')
    for seed, folder in MODEL_FOLDERS.items():
        p.add_argument(f'--seed{seed}-root', type=Path, default=MODEL_BASE / folder)
    p.add_argument('--selection-seed', type=int, default=20261110)
    p.add_argument('--partition-seed', type=int, default=20261111)
    p.add_argument('--noise-seed-base', type=int, default=202611120000)
    p.add_argument('--check-only', action='store_true', help='Validate without writing outputs')
    args = p.parse_args(argv)
    args.split, args.image_size = 'train', 64
    args.timesteps, args.repetitions = list(TIMESTEPS), REPETITIONS
    for key in ('selection_seed', 'partition_seed', 'noise_seed_base'):
        if not 0 <= getattr(args, key) < 2**63:
            p.error('Random seeds must be in [0, 2**63)')
    if args.noise_seed_base + 4 * REPETITIONS * N_PER_GROUP >= 2**63:
        p.error('Noise seed allocation exceeds supported range')
    return args


def load_previous(args, current_sources):
    path = args.previous_preparation_dir / 'manifest.json'
    previous = json.loads(path.read_text())
    if (previous.get('status') != 'complete' or previous.get('schema_version') != 2
            or previous.get('experiment') != 'forward_noise_cutoff_preparation'
            or previous.get('dataset_split') != 'train'):
        raise ValueError('Expected the completed original v2 training-split preparation manifest')
    prep = previous.get('preprocessing', {})
    if (prep.get('size') != [64, 64] or prep.get('resize') != 'Pillow bilinear'
            or prep.get('normalization') != 'uint8 / 127.5 - 1'
            or prep.get('random_augmentation') is not False):
        raise ValueError('Previous preprocessing differs; resolve before proceeding')
    indices = []
    for group in GROUPS:
        rows = previous['groups'][group]['records']
        if len(rows) != 50:
            raise ValueError('Previous preparation must have exactly 50 originals in each group')
        for row in rows:
            idx = row['dataset_index']
            if (type(idx) is not int or not 0 <= idx < 50000
                    or row['sample_id'] != f'train_{idx:05d}'):
                raise ValueError('Invalid previous training image identity')
            indices.append(idx)
    if len(set(indices)) != 100:
        raise ValueError('Previous groups must contain 100 distinct original IDs')
    for seed in MODEL_FOLDERS:
        old = previous['scheduler_sources'][str(seed)]
        new = current_sources[str(seed)]
        if old['selected_settings'] != new['selected_settings'] or old['betas_sha256'] != new['betas_sha256']:
            raise ValueError('Noise schedule differs from the previous study')
    return indices, {'path': str(path.resolve()), 'sha256': file_hash(path),
                     'excluded_sample_ids': [f'train_{i:05d}' for i in indices],
                     'dataset_file_sha256': sorted(r['sha256'] for r in previous['dataset_files']),
                     'previous_noise_seeds': [info['seed'] for row in previous['noise_plan']
                                              for info in row['banks'].values()]}


def balanced_fresh_groups(ds, excluded, selection_seed, partition_seed, n_per_group=N_PER_GROUP):
    """Deterministically skip exact image duplicates, including prior originals."""
    if n_per_group % 10:
        raise ValueError('Group size must be divisible by ten')
    labels = np.asarray(ds.targets)
    if set(np.unique(labels)) != set(range(10)):
        raise ValueError('Expected CIFAR-10 labels 0..9')
    seen_original, seen_prepared = set(), set()
    def content_keys(idx):
        im, _ = ds[idx]
        if im.mode != 'RGB' or im.size != (32, 32):
            raise ValueError('Expected RGB 32x32 originals')
        return (hashlib.sha256(im.tobytes()).hexdigest(),
                hashlib.sha256(im.resize((64,64), Image.Resampling.BILINEAR).tobytes()).hexdigest())
    for idx in excluded:
        raw, resized = content_keys(idx)
        seen_original.add(raw)
        seen_prepared.add(resized)
    rng = np.random.default_rng(selection_seed)
    partition = np.random.default_rng(partition_seed)
    groups = {g: [] for g in GROUPS}
    excluded_set, skipped = set(excluded), []
    per_class = n_per_group // 10
    for label in range(10):
        candidates = rng.permutation(np.flatnonzero(labels == label))
        chosen = []
        for idx in candidates:
            idx = int(idx)
            if idx in excluded_set:
                continue
            raw, resized = content_keys(idx)
            if raw in seen_original or resized in seen_prepared:
                skipped.append(idx)
                continue
            chosen.append(idx)
            seen_original.add(raw)
            seen_prepared.add(resized)
            if len(chosen) == 2 * per_class:
                break
        if len(chosen) != 2 * per_class:
            raise ValueError(f'Not enough distinct eligible images for class {label}')
        chosen = partition.permutation(chosen).tolist()
        groups['selection'].extend(chosen[:per_class])
        groups['evaluation'].extend(chosen[per_class:])
    groups = {g: partition.permutation(ids).tolist() for g, ids in groups.items()}
    return groups, skipped


def noise_seeds(base, group, repetition, bank, n=N_PER_GROUP):
    stream = (GROUPS.index(group) * REPETITIONS + repetition) * 2 + ('A', 'B').index(bank)
    return list(range(base + stream*n, base + (stream+1)*n))


def noise_for_seeds(seeds, image_size=64):
    """One CPU float32 randn(CHW) call per image: invariant to analysis batch size."""
    return torch.stack([make_noise((3, image_size, image_size), seed) for seed in seeds])


def run(args):
    dependencies()
    check_paths(args)
    out = args.out_dir.resolve()
    previous_root = args.previous_preparation_dir.resolve()
    if out.is_relative_to(previous_root) or previous_root.is_relative_to(out):
        raise ValueError('Output must be separate from previous preparation')
    scheduler, sources = load_schedules(args)
    if len(scheduler.betas) != 1000 or scheduler.config.beta_schedule != 'squaredcos_cap_v2':
        raise ValueError('Expected the original 1000-step cosine schedule')
    old_indices, previous_record = load_previous(args, sources)
    ds, dataset_files = load_dataset(args)
    if sorted(r['sha256'] for r in dataset_files) != previous_record['dataset_file_sha256']:
        raise ValueError('Dataset training batch hashes differ from the previous preparation')
    groups, skipped = balanced_fresh_groups(ds, old_indices, args.selection_seed,
                                           args.partition_seed, N_PER_GROUP)
    again, skipped_again = balanced_fresh_groups(ds, old_indices, args.selection_seed,
                                               args.partition_seed, N_PER_GROUP)
    if groups != again or skipped != skipped_again:
        raise RuntimeError('Image selection is not reproducible')
    all_indices = [i for g in GROUPS for i in groups[g]]
    if len(set(all_indices)) != 2*N_PER_GROUP or set(all_indices) & set(old_indices):
        raise RuntimeError('Image group identity overlap')
    prepared = {}
    for group in GROUPS:
        original, clean, records = prepare_clean(ds, groups[group], 64, 'train')
        counts = {name: sum(r['class_name'] == name for r in records) for name in ds.classes}
        if any(v != N_PER_GROUP//10 for v in counts.values()):
            raise RuntimeError('Class balance failed')
        for row in records:
            row['group'] = group
        prepared[group] = {'original': original, 'clean': clean, 'records': records,
                           'class_counts': counts}
    plan, all_seeds, bank_hashes = [], set(), set()
    for group in GROUPS:
        clean = prepared[group]['clean']
        for rep in range(REPETITIONS):
            for bank in ('A','B'):
                seeds = noise_seeds(args.noise_seed_base, group, rep, bank, N_PER_GROUP)
                if all_seeds.intersection(seeds) or set(seeds).intersection(previous_record['previous_noise_seeds']):
                    raise ValueError('Noise seeds overlap')
                all_seeds.update(seeds)
                epsilon = noise_for_seeds(seeds)
                digest = tensor_hash(epsilon)
                if digest in bank_hashes:
                    raise RuntimeError('Identical noise banks')
                bank_hashes.add(digest)
                probe = sorted(set([0, len(seeds)//2, len(seeds)-1]))
                repeat_noise = noise_for_seeds([seeds[i] for i in probe])
                if not torch.equal(epsilon[probe], repeat_noise):
                    raise RuntimeError('Noise changes when batch composition changes')
                anchors = {}
                for t in (0,500,999):
                    state = add_noise(scheduler, clean[probe], repeat_noise, t)
                    if not torch.equal(state, add_noise(scheduler, clean[probe], repeat_noise, t)):
                        raise RuntimeError('State repeatability failed')
                    anchors[str(t)] = tensor_hash(state)
                plan.append({'group': group, 'repetition': rep, 'bank': bank,
                             'seeds_in_saved_sample_order': seeds,
                             'noise_tensor_sha256': digest,
                             'probe_positions': probe, 'probe_state_sha256': anchors})
                del epsilon
            print(f'Checked {group}, noise pair {rep+1}/{REPETITIONS}', flush=True)
    logical_states = 2*N_PER_GROUP * 2*REPETITIONS*len(TIMESTEPS)
    raw_bytes = logical_states*3*64*64*4
    print(f'Preflight passed: {N_PER_GROUP} selection + {N_PER_GROUP} evaluation; '
          f'{N_PER_GROUP//10}/class/group; previous 100 IDs excluded.', flush=True)
    print(f'{len(TIMESTEPS)} timesteps; {REPETITIONS} A/B repetitions; '
          f'{logical_states:,} virtual noisy states (~{raw_bytes/1e9:.2f} GB if stored).', flush=True)
    print('CPU preparation only; lazy noise; no retrieval, FFT, model weights or evaluation scoring.', flush=True)
    if args.check_only:
        print('Check-only complete: no output files written.', flush=True)
        return 0
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    required = 2*N_PER_GROUP*(3*64*64*4 + 3*32*32) + 64*1024**2
    if shutil.disk_usage(parent).free < required:
        raise OSError('Insufficient space; allow at least 200 MB for default preparation')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema_version': 3, 'experiment': 'cifar10_cutoff_lazy_preparation',
        'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': file_hash(__file__), 'dataset_split': 'train',
        'arguments': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'dataset_files': dataset_files, 'previous_preparation': previous_record,
        'scheduler_sources': sources, 'timesteps': TIMESTEPS, 'repetitions': REPETITIONS,
        'n_noisy_image_states_virtual': logical_states, 'noisy_states_saved': False,
        'preprocessing': {'size': [64,64], 'resize': 'Pillow bilinear',
                          'normalization': 'uint8 / 127.5 - 1', 'random_augmentation': False},
        'skipped_duplicate_dataset_indices': skipped,
        'groups': {g: {'n_images': N_PER_GROUP, 'class_counts': prepared[g]['class_counts'],
                       'records': prepared[g]['records'],
                       'clean_tensor_sha256': tensor_hash(prepared[g]['clean']),
                       'original_tensor_sha256': tensor_hash(prepared[g]['original'])}
                   for g in GROUPS},
        'noise': {'plan_file': 'noise_plan.json', 'device': 'cpu', 'dtype': 'float32',
                  'algorithm': 'per_image_cpu_torch_randn_v1',
                  'recipe': 'For each saved seed: CPU Generator.manual_seed(seed); randn((3,64,64),float32). Stack in saved ID order.',
                  'coupling': 'Same epsilon across t per image/bank/repetition; not a Markov trajectory.',
                  'reconstruction': 'sqrt(alpha_bar[t])*clean + sqrt(1-alpha_bar[t])*epsilon; do not clip',
                  'compatibility': 'Regenerate on the same torch version and verify noise/probe hashes; fail if mismatched.'},
        'checks': {'disjoint_ids_and_excluded_previous_100': True,
                   'no_exact_original_or_prepared_duplicates_in_new_images_or_against_previous': True,
                   'balanced_groups_and_repeatable_selection': True,
                   'distinct_noise_seeds_and_bank_hashes': True,
                   'probe_noise_batch_invariance': True,
                   'probe_forward_equation_and_repeatability': True},
        'validation_scope': 'Every full noise bank hashed; three image positions at t=0,500,999 checked per bank. Other noisy states not materialized.',
        'holdout_policy': 'Preparation may read/validate both groups, but no evaluation previews or scores. Freeze selection choices before scoring evaluation.',
        'legacy_analysis_compatible': False, 'fft_or_similarity_metrics_computed': False,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch','torchvision','diffusers','numpy','Pillow')},
        'python_version': platform.python_version(), 'artifacts': [],
    }
    write_json(out/'manifest.json', manifest)
    def record(path):
        manifest['artifacts'].append({'file': str(path.relative_to(out)), 'sha256': file_hash(path)})
    def save_json(name, obj):
        path = out/name
        write_json(path, obj)
        record(path)
    try:
        save_json('noise_plan.json', plan)
        save_json('selected_images.json', [r for g in GROUPS for r in prepared[g]['records']])
        save_json('split.json', {g: [r['sample_id'] for r in prepared[g]['records']] for g in GROUPS})
        save_json('planned_protocol.json', {
            'status': 'prespecified_not_run', 'candidate_radii': RADII,
            'null_radius_means': 'full spectrum, all non-DC bins',
            'feature': 'grayscale(.299,.587,.114); subtract spatial mean; orthonormal FFT2; log1p(abs(FFT)**2); discard phase; exclude DC',
            'frequency_units': 'cycles per 64x64 prepared-image pixel',
            'mask': '0 < sqrt(fx**2+fy**2) <= R',
            'distance': 'RMS feature difference over included bins',
            'retrieval': 'exact original-ID top-1; both A-to-B and B-to-A; mean directions then repetitions',
            'distance_ties': {'atol': 1e-10, 'rtol': 1e-7, 'credit': '1/k if correct among k nearest tied candidates'},
            'selection': {'rule': 'maximize mean accuracy at each t; smallest R wins ties; full last',
                          'tie_atol': 1e-12, 'tie_rtol': 0, 'smoothing': 'none'},
            'evaluation_methods': ['frozen_selected_R(t)', 'fixed_R=0.1', 'full_spectrum'],
            'n_candidates_per_query': N_PER_GROUP, 'chance_reference': 1/N_PER_GROUP,
            'controls': ['clean vs clean', 'independent pure-noise A/B'],
            'dependence': 'Directions/repetitions reuse originals; timesteps reuse epsilon. Queries are not independent images.'})
        save_json('noise_levels.json', [
            {'timestep': t, 'alpha_bar': float(scheduler.alphas_cumprod[t]),
             'signal_scale': float(scheduler.alphas_cumprod[t].sqrt()),
             'noise_scale': float((1-scheduler.alphas_cumprod[t]).sqrt())} for t in TIMESTEPS])
        path = out/'schedule.pt'
        torch.save({'betas': scheduler.betas.cpu(),
                    'alphas_cumprod': scheduler.alphas_cumprod.cpu()}, path)
        record(path)
        for group, data in prepared.items():
            folder = out/group
            folder.mkdir()
            path = folder/'clean.pt'
            torch.save({'samples': data['clean'], 'original_uint8': data['original'],
                        'sample_ids': [r['sample_id'] for r in data['records']],
                        'records': data['records'], 'group': group}, path)
            record(path)
        folder = out/'selection'/'previews'
        folder.mkdir()
        data = prepared['selection']
        seeds = plan[0]['seeds_in_saved_sample_order'][:16]
        epsilon = noise_for_seeds(seeds)
        grid(data['clean'][:16], data['records'][:16], 'Selection clean reference', folder/'clean.png')
        record(folder/'clean.png')
        for t in (0,250,500,750,999):
            path = folder/f'bank_A_t_{t:04d}.png'
            grid(add_noise(scheduler,data['clean'][:16],epsilon,t), data['records'][:16],
                 f'Selection rep 0 bank A; t={t}', path)
            record(path)
        path = folder/'bank_A_pure_noise.png'
        grid(epsilon, data['records'][:16], 'Selection pure-noise reference', path)
        record(path)
        manifest['status'] = 'complete'
        manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out/'manifest.json', manifest)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        write_json(out/'manifest.json', manifest)
        raise
    print(f'Complete: {out}', flush=True)
    print('Saved clean images, schedule, seeds, hashes and selection previews. '
          'No FFT, R selection, matching or evaluation scoring performed.', flush=True)
    print('Use the forthcoming v3 validation/analysis scripts, NOT the old v2 scripts.', flush=True)
    return 0


def main(argv=None):
    return run(parse_args(argv))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)

