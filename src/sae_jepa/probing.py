"""Sparse probes on frozen stage-2 features, with train-only feature selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from .data import write_json
from .evaluate import _autocast
from .probe_data import (DATASETS, SPLITS, checkpoint_info, collect, file_hash,
                         prepare, read_tasks, text_id)
from .stage2 import FORMAT, build_front, config, frontend_identity, tensor_hash
from .topk import TopKSAE


def select_features(x, labels, k):
    if x.ndim != 2 or not 1 <= k <= x.shape[1] or set(labels.tolist()) != {0, 1}:
        raise ValueError("feature selection requires binary train labels and a valid k")
    scores = (x[labels == 1].double().mean(0) - x[labels == 0].double().mean(0)).abs()
    return torch.argsort(scores, descending=True, stable=True)[:k]


def accuracy(logits, labels):
    predicted = (logits >= 0).long()
    correct = predicted == labels
    return {"accuracy": float(correct.double().mean()),
            "balanced_accuracy": sum(float(correct[labels == c].double().mean()) for c in [0, 1]) / 2,
            "count": len(labels)}


def fit_logistic(x, labels, c, max_iter):
    """Deterministic CPU float64 convex logistic probe, zero initialization."""
    x, labels = x.double(), labels.double()
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=max_iter, tolerance_grad=1e-8,
                                tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, labels) + w.square().sum() / (2 * c * len(x))
        loss.backward()
        return loss
    optimizer.step(closure)
    closure()
    grad = max(float(w.grad.abs().max()), float(b.grad.abs()))
    if not all(torch.isfinite(v).all() for v in [w, b, w.grad, b.grad]):
        raise ValueError("nonfinite logistic fit")
    return w.detach(), b.detach(), {"max_abs_gradient": grad,
        "iterations": int(optimizer.state[w].get("n_iter", 0)), "max_iter": max_iter}


def fit_task(features, rows, ks, cs, max_iter, standardize=False, include_test=False):
    # Selection/centering/training never receives validation or test labels.
    split = {}
    for name in SPLITS:
        selected = [r for r in rows if r['split'] == name]
        split[name] = (torch.stack([features[text_id(r['text'])] for r in selected]),
                       torch.tensor([r['label'] for r in selected]))
    train, labels = split['train']
    results = {}
    for k in ks:
        indices = select_features(train, labels, k)
        x = train[:, indices].double()
        mean = x.mean(0) if standardize else torch.zeros(k, dtype=torch.float64)
        scale = x.std(0, correction=0).clamp_min(1e-8) if standardize else torch.ones(k, dtype=torch.float64)
        x = (x - mean) / scale
        vx, vy = split['validation']; vx = (vx[:, indices].double() - mean) / scale
        candidates = []
        best = None
        for c in sorted(cs):
            w, b, diagnostics = fit_logistic(x, labels, c, max_iter)
            score = accuracy(vx @ w + b, vy)
            candidates.append({"C": c, "validation": score, "fit": diagnostics})
            # Ties prefer stronger regularization; no test-dependent choices.
            if best is None or score['accuracy'] > best[0]:
                best = (score['accuracy'], w, b, c, diagnostics)
        _, w, b, c, diagnostics = best
        result = {"feature_indices": indices.tolist(), "C": c,
            "train": accuracy(x @ w + b, labels), "validation": accuracy(vx @ w + b, vy),
            "selection": "train absolute class mean difference; stable index tie-break",
            "candidates": candidates, "fit": diagnostics, "weights": w.tolist(), "bias": float(b),
            "mean": mean.tolist(), "scale": scale.tolist()}
        if include_test:
            tx, ty = split['test']; tx = (tx[:, indices].double() - mean) / scale
            result['test'] = accuracy(tx @ w + b, ty)
        results[str(k)] = result
    return results


@torch.no_grad()
def pooled_features(state, cache, device, token_batch_size):
    """Encode each token before pooling, never encode a mean residual."""
    cfg = config(state['config'])
    front = build_front(state['frontend'], device)
    with torch.random.fork_rng(devices=[]):
        sae = TopKSAE(front.cfg.d_latent, cfg.dictionary_size, cfg.k)
    sae.load_state_dict(state['sae']); sae.to(device).eval().requires_grad_(False)
    mean = state['calibration']['mean'].to(device)
    scale = state['calibration']['scale']
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    features = {}
    for chunk in manifest['chunks']:
        path = cache / chunk['file']
        if file_hash(path) != chunk['sha256']:
            raise ValueError("activation cache checksum mismatch")
        batch = torch.load(path, map_location='cpu', weights_only=True)
        if batch['h'].ndim != 3 or batch['mask'].shape != batch['h'].shape[:2]:
            raise ValueError("invalid activation/mask shapes")
        for key, h, mask in zip(batch['ids'], batch['h'], batch['mask']):
            if key in features or not mask.any():
                raise ValueError("duplicate or empty activation example")
            total = torch.zeros(cfg.dictionary_size, device=device, dtype=torch.float64)
            values = h[mask.bool()]
            for part in values.split(token_batch_size):
                with _autocast(device, cfg.amp_dtype):
                    y = front.encode_dense(part.to(device)).float()
                    z = sae.encode((y - mean) / scale)
                if not torch.isfinite(z).all():
                    raise ValueError("nonfinite SAE activations")
                total += z.double().sum(0)
            features[key] = (total / len(values)).float().cpu()
    if set(features) != set(manifest['ids']):
        raise ValueError("activation cache IDs are incomplete")
    return features


def aggregate(tasks, split, k):
    datasets = defaultdict(list)
    for record in tasks.values():
        datasets[record['dataset']].append(record['scores'][str(k)][split]['accuracy'])
    per_dataset = {name: statistics.mean(scores) for name, scores in datasets.items()}
    return {"dataset_macro_accuracy": statistics.mean(per_dataset.values()),
            "task_macro_accuracy": statistics.mean(v for scores in datasets.values() for v in scores),
            "per_dataset_accuracy": per_dataset}


def evaluate_checkpoints(args):
    if (not args.ks or 1 not in args.ks or len(set(args.ks)) != len(args.ks)
            or any(k < 1 for k in args.ks) or not args.cs
            or any(not math.isfinite(c) or c <= 0 for c in args.cs)
            or args.max_iter < 1 or args.token_batch_size < 1):
        raise ValueError("require unique positive ks including 1, positive Cs and iteration/batch limits")
    rows = read_tasks(args.tasks)
    cache = Path(args.activations)
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'sae-jepa-probe-activations-v1':
        raise ValueError("not a probe activation cache")
    if manifest['signature']['tasks_sha256'] != file_hash(args.tasks):
        raise ValueError("task file differs from activation cache")
    if set(manifest['ids']) != {text_id(r['text']) for r in rows}:
        raise ValueError("task/cache IDs differ")
    for path in args.checkpoints:
        if checkpoint_info(path) != manifest['signature']['identity']:
            raise ValueError("checkpoint model, layer or exclusion policy differs from cache")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['task']].append(row)
    summary = []
    baseline = None
    for i, path in enumerate(args.checkpoints):
        print(f"Probing checkpoint {i+1}/{len(args.checkpoints)}: {path}", flush=True)
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        comparison = (state['config'], state['step'], state['initial_sha256'])
        if baseline is not None and comparison != baseline:
            raise ValueError("stage-2 settings, steps or initialization differ across candidates")
        baseline = comparison
        if max(args.ks) > state['config']['dictionary_size']:
            raise ValueError("probe k exceeds dictionary size")
        features = pooled_features(state, cache, torch.device(args.device), args.token_batch_size)
        result = {"checkpoint": path, "sae_sha256": tensor_hash(state['sae']),
            "frontend_sha256": tensor_hash(state['frontend']['model']),
            **frontend_identity(state['frontend']),
            "step": state['step'], "tasks": {}}
        for task, items in sorted(grouped.items()):
            result['tasks'][task] = {"dataset": items[0]['dataset'],
                "scores": fit_task(features, items, args.ks, args.cs, args.max_iter,
                                   args.standardize, args.include_test)}
        result['aggregate'] = {s: {str(k): aggregate(result['tasks'], s, k) for k in args.ks}
                               for s in (['validation', 'test'] if args.include_test else ['validation'])}
        write_json(output / f'model-{i:02d}.json', result)
        summary.append({k: v for k, v in result.items() if k != 'tasks'})
        del features, state
    write_json(output / 'comparison.json', {"arguments": vars(args), "tasks_sha256": file_hash(args.tasks),
        "activation_manifest_sha256": file_hash(cache / 'manifest.json'),
        "protocol": "SAEBench-style mean-difference sparse probes; held-out validation for C; not exact paper reproduction",
        "models": summary})
    lines = ['# Sparse probing (primary metric: Top-1)', '',
             'Validation selects C. Test is evaluated only with --include-test. No test-based feature selection.', '',
             '| Model | Front-end | lambda | beta | split | probe features | Dataset macro accuracy | Task macro accuracy |',
             '|---|---|---:|---:|---|---:|---:|---:|']
    for i, result in enumerate(summary):
        for split, scores in result['aggregate'].items():
            for k, r in scores.items():
                lines.append(f"| {i:02d} | {result['frontend_name']} | {result['frontend_lambda']:g} | "
                             f"{result['frontend_covariance_weight']:g} | {split} | {k} | "
                             f"{r['dataset_macro_accuracy']:.4f} | {r['task_macro_accuracy']:.4f} |")
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True)
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--train-size', type=int, default=4000)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--validation-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('collect')
    p.add_argument('--tasks', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model'); p.add_argument('--revision')
    p.add_argument('--context-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('evaluate')
    p.add_argument('--tasks', required=True); p.add_argument('--activations', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True); p.add_argument('--output', required=True)
    p.add_argument('--ks', type=int, nargs='+', default=[1, 2, 5])
    p.add_argument('--cs', type=float, nargs='+', default=[.1, 1., 10.])
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--token-batch-size', type=int, default=512)
    p.add_argument('--standardize', action='store_true')
    p.add_argument('--include-test', action='store_true')
    p.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    return {'prepare': prepare, 'collect': collect, 'evaluate': evaluate_checkpoints}[args.command](args)


if __name__ == '__main__':
    main()
