"""Frozen full-dimensional ZCA input preconditioning, fitted on train only."""
import argparse
import math
import random
from pathlib import Path

import torch

from .data import write_json
from .normalization import (_source, _source_arguments, check_normalization_matches,
                            load_normalization, save_normalization)

FORMAT = 'sae-jepa-input-zca-v1'


@torch.no_grad()
def fit(source, normalization, *, epsilon=1e-4, maximum_positions=0, chunk_size=2048, device='cpu', sample_seed=1729):
    check_normalization_matches(normalization, source)
    if not math.isfinite(epsilon) or epsilon <= 0 or maximum_positions < 0 or chunk_size < 1:
        raise ValueError('epsilon must be positive; invalid sampling/chunk settings')
    device = torch.device(device)
    d = source.d_in
    mean = torch.zeros(d, dtype=torch.float64, device=device)
    m2 = torch.zeros(d, d, dtype=torch.float64, device=device)
    mu = normalization['mean'].to(device=device, dtype=torch.float64)
    scale = float(normalization['scale'])
    count = 0
    used = []
    entries = source.paths('train')
    selected = None
    counts = [int(source.sequence_counts(e).sum()) for e in entries]
    if maximum_positions:
        # Uniform sample over all usable train positions, without allocating a
        # dataset-sized randperm or reading held-out activation values.
        selected = sorted(random.Random(sample_seed).sample(range(sum(counts)), min(maximum_positions, sum(counts))))
    cursor = offset = 0
    for entry, available in zip(entries, counts):
        if selected is None:
            batches = source.positions(entry).split(chunk_size)
        else:
            start = cursor
            while cursor < len(selected) and selected[cursor] < offset + available:
                cursor += 1
            local = torch.tensor(selected[start:cursor], dtype=torch.long) - offset
            batches = (source.gather(entry, indices) for indices in local.split(chunk_size) if len(indices))
        offset += available
        used_count = 0
        for batch in batches:
            if len(batch) == 0:
                continue
            x = (batch.to(device=device, dtype=torch.float64) - mu) / scale
            if not torch.isfinite(x).all():
                raise ValueError('nonfinite train activations')
            n = len(x)
            local_mean = x.mean(0)
            centered = x-local_mean
            delta = local_mean-mean
            m2 += centered.T @ centered + torch.outer(delta, delta) * (count*n/(count+n))
            mean += delta * (n/(count+n))
            count += n
            used_count += n
        if used_count:
            used.append({'entry': entry, 'count': used_count})
    if count < 2:
        raise ValueError('whitening requires at least two train samples')
    eigenvalues, u = torch.linalg.eigh(m2/count)
    eigenvalues = eigenvalues.clamp_min(0)
    matrix = (u * (eigenvalues+epsilon).rsqrt()[None, :]) @ u.T
    return {'format': FORMAT, 'split': 'train', 'mean': normalization['mean'].clone(), 'scale': scale,
            'shards': source.paths('train'), 'burn_in_excluded': source.burn_in,
            'manifest_fingerprint': source.fingerprint, 'd_in': d, 'count': count,
            'used_shards': used, 'maximum_positions': maximum_positions,
            'sampling': 'all usable train positions' if not maximum_positions else 'uniform without replacement over usable train positions',
            'sample_seed': sample_seed, 'available_train_positions': sum(counts),
            'epsilon': epsilon, 'center': mean.cpu().float(), 'matrix': matrix.cpu().float(),
            'train_eigenvalues': eigenvalues.flip(0).cpu(),
            'expected_transformed_eigenvalues': (eigenvalues/(eigenvalues+epsilon)).flip(0).cpu(),
            'convention': 'x=(h-mu)/s; z=(x-center) U diag((eigenvalues+epsilon)^-1/2) U^T; no dimension reduction'}


def validate(stats, source, normalization):
    if stats.get('format') != FORMAT or stats.get('split') != 'train':
        raise ValueError('input whitening must be train-fit ZCA statistics')
    if not math.isfinite(stats['epsilon']) or stats['epsilon'] <= 0 or stats['count'] < 2:
        raise ValueError('invalid input whitening epsilon/count')
    check_normalization_matches(stats, source)
    if not torch.equal(stats['mean'], normalization['mean']) or stats['scale'] != normalization['scale']:
        raise ValueError('input whitening scalar normalization differs')
    d = source.d_in
    if stats['matrix'].shape != (d, d) or stats['center'].shape != (d,):
        raise ValueError('input whitening shape mismatch')
    if not torch.isfinite(stats['matrix']).all() or not torch.isfinite(stats['center']).all():
        raise ValueError('nonfinite input whitening statistics')


def install(model, stats):
    device = model.input_mean.device
    model.whitening_epsilon = stats['epsilon']
    model.whitening_count = stats['count']
    for name, key in [('whitening_matrix', 'matrix'), ('whitening_center', 'center')]:
        model.register_buffer(name, stats[key].detach().to(device=device, dtype=torch.float32).clone())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    _source_arguments(p)
    p.add_argument('--normalization', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epsilon', type=float, default=1e-4)
    p.add_argument('--maximum-positions', type=int, default=0)
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--sample-seed', type=int, default=1729)
    p.add_argument('--device', default='cpu')
    args = p.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise ValueError('whitening output already exists; choose a new path')
    source = _source(args)
    stats = fit(source, load_normalization(args.normalization, source), epsilon=args.epsilon,
                maximum_positions=args.maximum_positions, chunk_size=args.chunk_size, device=args.device,
                sample_seed=args.sample_seed)
    save_normalization(stats, output)
    report = {k: v for k, v in stats.items() if not isinstance(v, torch.Tensor)}
    report.update(train_eigenvalues=stats['train_eigenvalues'].tolist(),
                  expected_transformed_eigenvalues=stats['expected_transformed_eigenvalues'].tolist())
    write_json(output.with_suffix('.json'), report)
    print(f"Train-fit ZCA: {stats['count']} positions, epsilon={args.epsilon}; {output}")


if __name__ == '__main__':
    main()
