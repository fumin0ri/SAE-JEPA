"""Trace paired-norm samples to tokens and independently verify stored activations.

Inspection requires the original shards, not an LLM. Recompute is opt-in and
replays entire stored sequences, never a cropped context or retokenized text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re

import torch

from .config import config_from_dict
from .data import DataSource, LEJEPA_FORMAT, parse_entry
from .diagnose import check_source
from .probe_data import blocks


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare(actual, reference):
    a, b = actual.detach().cpu().double().flatten(), reference.detach().cpu().double().flatten()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('activation shapes differ or contain nonfinite values')
    error = (a - b).norm()
    denominator = a.norm() * b.norm()
    return {'exact_equal': torch.equal(a, b), 'max_abs_error': float((a-b).abs().max()),
            'relative_l2_error': float(error / b.norm()) if b.norm() > 0 else None,
            'cosine': float((a @ b / denominator).clamp(-1, 1)) if denominator > 0 else None,
            'actual_sqnorm_per_dim': float(a.square().mean()),
            'reference_sqnorm_per_dim': float(b.square().mean())}


class StoredSequences:
    """Official safetensors reader, independent of DataSource's memmap reader."""
    def __init__(self, source):
        self.source = source
        raw = json.loads(source.manifest_path.read_text(encoding='utf-8'))
        if source.format != LEJEPA_FORMAT:
            raise ValueError('verification currently requires LeJEPA-SAE flat safetensors shards')
        self.shards = {s['file']: s for s in raw['shards']}

    def token_inventory(self, rows):
        """Read token arrays one shard at a time; never scan activation arrays."""
        from safetensors import safe_open
        grouped = defaultdict(list)
        for i, row in enumerate(rows):
            grouped[parse_entry(row['shard'])[0]].append((i, row))
        result = {}
        for path, items in grouped.items():
            with safe_open(str(self.source.root / path), framework='pt', device='cpu') as f:
                ids = f.get_tensor('token_ids') if 'token_ids' in f.keys() else None
                if ids is not None and (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64)):
                    raise ValueError('token_ids must be a flat integer tensor')
                for i, row in items:
                    _, start, stop = parse_entry(row['shard'])
                    sequences = self.shards[path]['sequences']
                    seq, pos = row['sequence'], row['position']
                    if not start <= seq < (len(sequences) if stop is None else stop):
                        raise ValueError('sequence outside shard entry range')
                    record = sequences[seq]
                    if not 0 <= pos < record['length']:
                        raise ValueError('position outside stored sequence')
                    result[i] = int(ids[record['offset'] + pos]) if ids is not None else None
        return result

    def read(self, row, full=False):
        from safetensors import safe_open
        entry = row['shard']
        path, start, stop = parse_entry(entry)
        seq, pos = row['sequence'], row['position']
        shard = self.shards[path]
        if not start <= seq < (len(shard['sequences']) if stop is None else stop):
            raise ValueError('sequence outside shard entry range')
        record = shard['sequences'][seq]
        offset, length = int(record['offset']), int(record['length'])
        if not 0 <= pos < length:
            raise ValueError('position outside stored sequence')
        with safe_open(str(self.source.root / path), framework='pt', device='cpu') as f:
            a = f.get_slice('activations')
            if len(a.get_shape()) != 2 or a.get_shape()[1] != self.source.d_in or offset + length > a.get_shape()[0]:
                raise ValueError('activation shape/sequence bounds do not match manifest')
            stored = a[offset + pos:offset + pos + 1][0].clone()
            inputs = {}
            for key in ('token_ids', 'attention_mask', 'position_ids'):
                if key in f.keys():
                    tensor = f.get_slice(key)
                    if tensor.get_shape() != [a.get_shape()[0]]:
                        raise ValueError(f'{key} must be a flat vector aligned with activations')
                    # Token identities are needed even during inspection.
                    if full or key == 'token_ids':
                        inputs[key] = tensor[offset:offset + length].clone().long()
        return stored, inputs, record

    def loader_row(self, row):
        entry = row['shard']
        _, start, _ = parse_entry(entry)
        seq, pos = row['sequence'], row['position']
        counts = self.source.sequence_counts(entry)
        if pos < self.source.burn_in:
            raise ValueError('diagnostic row points to an excluded position')
        local = int(counts[:seq-start].sum()) + pos - self.source.burn_in
        location = torch.tensor([local])
        got_seq, got_pos = self.source.locate(entry, location)
        if int(got_seq[0]) != seq or int(got_pos[0]) != pos:
            raise ValueError('loader metadata round-trip mismatch')
        return self.source.gather(entry, location)[0]


def capture(model, inputs, layer):
    stack = blocks(model)
    if not 0 <= layer < len(stack):
        raise ValueError('block_output layer index out of range')
    captured = []
    def hook(_module, _args, output):
        captured.append((output[0] if isinstance(output, tuple) else output).detach().cpu())
    handle = stack[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].ndim != 3 or captured[0].shape[0] != 1:
        raise ValueError('hook did not capture exactly one [1, tokens, hidden] tensor')
    return captured[0][0]


def replay_inputs(inputs, policy, device):
    if 'token_ids' not in inputs:
        raise ValueError('token_ids missing; original model input cannot be reconstructed')
    ids = inputs['token_ids']
    if (ids < 0).any():
        raise ValueError('negative token ID')
    if policy == 'stored':
        if 'attention_mask' not in inputs:
            raise ValueError('attention_mask missing; use --attention-mask all-ones only if extraction used unpadded sequences')
        mask = inputs['attention_mask']
    else:
        mask = torch.ones_like(ids)
    if not ((mask == 0) | (mask == 1)).all() or not mask.any():
        raise ValueError('attention mask must be binary and nonempty')
    result = {'input_ids': ids[None].to(device), 'attention_mask': mask[None].to(device)}
    if 'position_ids' in inputs:
        result['position_ids'] = inputs['position_ids'][None].to(device)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True, help='paired norm-summary JSON')
    p.add_argument('--samples', help='defaults to samples_path beside summary')
    p.add_argument('--checkpoint', help='defaults to checkpoint in summary provenance')
    p.add_argument('--activation-manifest', help='relocate original activation data')
    p.add_argument('--output', required=True, help='new/empty directory')
    p.add_argument('--min-output-energy', type=float, default=10., help='select y_sqnorm_per_dim above this value')
    p.add_argument('--max-outliers', type=int, default=64)
    p.add_argument('--controls', type=int, default=8, help='ordinary controls; deterministic random sample')
    p.add_argument('--matched-per-token', type=int, default=2, help='ordinary controls of each outlier token ID')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--context-radius', type=int, default=16)
    p.add_argument('--tokenizer', help='optional model/tokenizer directory for decoding, no retokenization')
    p.add_argument('--recompute', action='store_true')
    p.add_argument('--model', help='optional local copy/model ID; defaults to manifest model')
    p.add_argument('--revision', help='defaults to recorded model revision; record any override')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    p.add_argument('--device', default='cuda')
    p.add_argument('--attention-mask', choices=['stored', 'all-ones'], default='stored')
    p.add_argument('--attn-implementation', choices=['eager', 'sdpa', 'flash_attention_2'], default='eager')
    p.add_argument('--allow-download', action='store_true', help='otherwise require locally cached model/tokenizer')
    return p.parse_args(argv)


def run(args):
    if (not math.isfinite(args.min_output_energy) or args.min_output_energy <= 0 or args.max_outliers < 1
            or args.controls < 0 or args.matched_per_token < 0 or args.context_radius < 0):
        raise ValueError('invalid selection parameters')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('output directory must be empty')
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    sample_path = Path(args.samples) if args.samples else summary_path.parent / summary['samples_path']
    rows = [json.loads(s) for s in sample_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    provenance = summary['provenance']
    checkpoint = args.checkpoint or provenance['checkpoint']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = config_from_dict(state['config'])
    source = DataSource(args.activation_manifest or cfg.data.activation_manifest,
        skip_burn_in=cfg.data.skip_burn_in, skip_leading_positions=cfg.data.skip_leading_positions,
        test_split=cfg.data.test_split, holdout_test_fraction=cfg.data.holdout_test_fraction)
    check_source(state, source)
    if source.fingerprint != provenance['data_fingerprint'] or state['step'] != provenance['step']:
        raise ValueError('norm diagnostic provenance differs from checkpoint/data')
    del state
    if len(rows) != summary['count'] or len(rows) < 2:
        raise ValueError('sample count differs from norm summary')
    entries = source.paths(provenance['split'])
    for i, row in enumerate(rows):
        if (row['sample_index'] != i or not 0 <= row['entry'] < len(entries)
                or row['shard'] != entries[row['entry']]):
            raise ValueError('sample identity or shard mapping mismatch')
        for k in ['h_sqnorm_per_dim', 'x_sqnorm_per_dim', 'y_sqnorm_per_dim']:
            if not math.isfinite(row[k]) or row[k] < 0:
                raise ValueError('invalid norm value in samples')
    reader = StoredSequences(source)
    scores = torch.tensor([r['y_sqnorm_per_dim'] for r in rows], dtype=torch.float64)
    ranked = torch.argsort(scores, descending=True, stable=True).tolist()
    candidates = [i for i in ranked if scores[i] > args.min_output_energy]
    selected = {i: ['outlier'] for i in candidates[:args.max_outliers]}
    ordinary = [i for i in range(len(rows)) if scores[i] <= args.min_output_energy]
    permutation = torch.randperm(len(ordinary), generator=torch.Generator().manual_seed(args.seed)).tolist()
    for j in permutation[:args.controls]:
        selected.setdefault(ordinary[j], []).append('ordinary')
    # Inspect all sampled identities so token frequencies have a denominator.
    identities = reader.token_inventory(rows)
    token_counts, outlier_tokens = Counter(), Counter()
    for i, row in enumerate(rows):
        token = identities[i]
        if token is not None:
            token_counts[token] += 1
            if scores[i] > args.min_output_energy:
                outlier_tokens[token] += 1
    targets = {identities[i] for i in selected if 'outlier' in selected[i]} - {None}
    matched = Counter()
    for j in permutation:
        i = ordinary[j]
        token = identities[i]
        if token in targets and matched[token] < args.matched_per_token:
            selected.setdefault(i, []).append('same_token_control')
            matched[token] += 1
    report = {'status': 'inspection_complete', 'source': source.record(), 'checkpoint': str(checkpoint),
        'norm_summary_sha256': file_hash(summary_path), 'norm_samples_sha256': file_hash(sample_path),
        'selection': vars(args), 'sample_count': len(rows), 'outlier_count': len(candidates),
        'selected_outliers': min(len(candidates), args.max_outliers), 'selected_count': len(selected),
        'missing_token_id_count': sum(t is None for t in identities.values()),
        'token_counts': [{'token_id': t, 'outlier_count': n, 'sample_count': token_counts[t],
                          'ordinary_count': token_counts[t]-n, 'matched_controls': matched[t]}
                         for t, n in outlier_tokens.most_common()],
        'limitations': ['Observed token association is not proof of causation.',
            'A recomputation mismatch does not by itself prove corruption.',
            'New round-trip checks test current serialization, not historical extraction.',
            'Original precision, attention backend, padding and position policies may be unrecorded.'],
        'samples': []}
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision or source.manifest.get('resolved_model_revision'),
                                                local_files_only=not args.allow_download, trust_remote_code=False)
    for i, groups in selected.items():
        row = rows[i]
        stored, inputs, record = reader.read(row, full=True)
        loader = reader.loader_row(row)
        norm = float(stored.double().square().mean())
        result = {**row, 'groups': groups, 'direct_vs_loader': compare(loader, stored),
            'logged_h_norm_matches': math.isclose(norm, row['h_sqnorm_per_dim'], rel_tol=1e-6, abs_tol=1e-8),
            'document_id': record.get('document_id'), 'segment_index': record.get('segment_index'),
            'sequence_length': int(record['length']), 'token_id': identities[i],
            'stored_attention_mask': 'attention_mask' in inputs, 'stored_position_ids': 'position_ids' in inputs}
        if 'token_ids' in inputs:
            pos = row['position']
            lo, hi = max(0, pos-args.context_radius), min(len(inputs['token_ids']), pos+args.context_radius+1)
            context = inputs['token_ids'][lo:hi].tolist()
            result.update(context_start_position=lo, context_token_ids=context,
                          token_ids_sha256=hashlib.sha256(inputs['token_ids'].numpy().tobytes()).hexdigest())
            if tokenizer is not None:
                result.update(token_text=tokenizer.decode([identities[i]], skip_special_tokens=False),
                    context_text=tokenizer.decode(context, skip_special_tokens=False),
                    is_special_token=identities[i] in tokenizer.all_special_ids)
        else:
            result['context_status'] = 'token_ids missing from shard'
        report['samples'].append(result)
    report['loader_matches_all_selected'] = all(r['direct_vs_loader']['exact_equal'] for r in report['samples'])
    write_json(out / 'verification.json', report)
    if args.recompute:
        try:
            recompute(args, source, reader, report, out)
            report['status'] = 'recomputation_complete'
        except Exception as error:
            report.update(status='recomputation_incomplete', error=f'{type(error).__name__}: {error}')
            write_json(out / 'verification.json', report)
            raise
    write_json(out / 'verification.json', report)
    return report


def recompute(args, source, reader, report, out):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModel
    recorded = source.manifest
    layer = recorded.get('layer')
    if type(layer) is not int or recorded.get('hook_point') != f'block_output:{layer}':
        raise ValueError('only explicit zero-based block_output:<layer> hooks are supported')
    for row in report['samples']:
        if not row['direct_vs_loader']['exact_equal'] or not row['logged_h_norm_matches']:
            raise ValueError('stored data/loader/logged norms disagree; resolve data identity before replay')
        _, inputs, _ = reader.read(row, full=True)
        replay_inputs(inputs, args.attention_mask, 'cpu')
    model_id = args.model or recorded.get('model')
    revision = args.revision or recorded.get('resolved_model_revision')
    if not model_id:
        raise ValueError('model ID missing')
    model = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation, local_files_only=not args.allow_download,
        trust_remote_code=False).to(args.device).eval()
    report['replay'] = {'model': model_id, 'revision_requested': revision,
        'revision_resolved': getattr(model.config, '_commit_hash', None),
        'recorded_model': recorded.get('model'), 'recorded_revision': recorded.get('resolved_model_revision'),
        'recorded_revision_is_immutable': bool(re.fullmatch('[0-9a-fA-F]{40}', recorded.get('resolved_model_revision') or '')),
        'dtype': args.dtype, 'device': args.device, 'attention_mask': args.attention_mask,
        'position_ids': 'stored when present; otherwise model default', 'attn_implementation': args.attn_implementation,
        'hook': recorded['hook_point'], 'batch_size': 1,
        'torch_version': torch.__version__, 'transformers_version': importlib.metadata.version('transformers')}
    grouped = defaultdict(list)
    for row in report['samples']:
        grouped[(row['shard'], row['sequence'])].append(row)
    for number, samples in enumerate(grouped.values()):
        _, inputs, _ = reader.read(samples[0], full=True)
        captured = capture(model, replay_inputs(inputs, args.attention_mask, args.device), layer)
        if captured.shape != (len(inputs['token_ids']), source.d_in):
            raise ValueError('replayed activation shape differs from stored sequence')
        for row in samples:
            stored, _, _ = reader.read(row)
            fresh = captured[row['position']]
            row['recomputed_vs_stored'] = compare(fresh, stored)
            quantized = fresh.to(stored.dtype).contiguous()
            row['storage_cast_vs_stored'] = compare(quantized, stored)
            # Round-trip a new diagnostic artifact; never rewrite original shards.
            path = out / f"roundtrip-{row['sample_index']:06d}.safetensors"
            save_file({'activations': quantized[None]}, str(path))
            with safe_open(str(path), framework='pt', device='cpu') as f:
                restored = f.get_tensor('activations')[0]
            row['roundtrip_file'] = path.name
            row['roundtrip_sha256'] = file_hash(path)
            row['new_roundtrip'] = compare(restored, quantized)
        write_json(out / 'verification.json', report)
        print(f'replayed {number+1}/{len(grouped)} sequences', flush=True)
    report['group_errors'] = {}
    for group in ('outlier', 'ordinary', 'same_token_control'):
        values = [r['recomputed_vs_stored']['relative_l2_error'] for r in report['samples'] if group in r['groups']]
        values = [v for v in values if v is not None]
        if values:
            report['group_errors'][group] = {'count': len(values), 'median_relative_l2': float(torch.tensor(values).median()),
                                            'maximum_relative_l2': max(values)}


def main(argv=None):
    report = run(parse_args(argv))
    print(f"{report['status']}: {report['selected_count']} samples; see verification.json")


if __name__ == '__main__':
    main()
