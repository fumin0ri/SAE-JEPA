"""Explicit binary task splits and shared block-output caches for sparse probes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import unicodedata

import torch

from .data import write_json
from .stage2 import FORMAT

SPLITS = ("train", "validation", "test")
DATASETS = ["LabHC/bias_in_bios_class_set1", "LabHC/bias_in_bios_class_set2",
            "LabHC/bias_in_bios_class_set3", "canrager/amazon_reviews_mcauley_1and5",
            "canrager/amazon_reviews_mcauley_1and5_sentiment", "codeparrot/github-code",
            "fancyzhx/ag_news", "Helsinki-NLP/europarl"]


def text_id(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tasks(path):
    # Split on "\n" only: str.splitlines() also breaks on U+2028, U+0085 etc., which
    # json.dumps(ensure_ascii=False) leaves unescaped inside text (e.g. europarl).
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise ValueError("empty probe task file")
    seen, groups, contents = set(), {}, {}
    tasks = {}
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in ["id", "group", "task", "dataset", "text"]):
            raise ValueError("every row requires nonempty id, group, task, dataset and text strings")
        if row.get("split") not in SPLITS or type(row.get("label")) is not int or row["label"] not in [0, 1]:
            raise ValueError("require explicit train/validation/test splits and binary integer labels")
        key = (row["task"], row["id"])
        if key in seen:
            raise ValueError("duplicate example ID within a task")
        seen.add(key)
        for registry, value in [(groups, row["group"]), (contents, text_id(row["text"]))]:
            key = (row["task"], value)
            if key in registry and registry[key] != row["split"]:
                raise ValueError("group or duplicate text crosses probe splits")
            if registry is contents and key in registry:
                raise ValueError("duplicate normalized text within a task")
            registry[key] = row["split"]
        tasks.setdefault(row["task"], []).append(row)
    for task, items in tasks.items():
        if len({r['dataset'] for r in items}) != 1:
            raise ValueError(f"{task}: dataset name is inconsistent")
        for split in SPLITS:
            if {r["label"] for r in items if r["split"] == split} != {0, 1}:
                raise ValueError(f"{task}: {split} must contain both classes")
    return rows


def truncated_token_key(tokenizer, context_length):
    """Dedup key matching collect(): texts whose truncated token ids agree are duplicates."""
    def key(text):
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=context_length)['input_ids']
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return key


def binary_rows(dataset, train, test, classes, seed, validation_fraction, dedup_key=text_id):
    """Deduplicate before splitting; reserve validation from the source train set."""
    rng = random.Random(seed)
    pools = {s: {} for s in SPLITS}
    seen = set()

    def add(text, unique, claimed):
        # Always dedup by normalized text; an extra key (e.g. truncated tokens) also
        # drops later texts that collide with an earlier, different text.
        key = text_id(text)
        if key in seen:
            return
        if dedup_key is not text_id:
            extra = "extra:" + dedup_key(text)
            if extra in seen or claimed.setdefault(extra, key) != key:
                return
        unique[key] = text

    for cls in classes:
        unique, claimed = {}, {}
        for text in train[cls]:
            add(text, unique, claimed)
        seen.update(unique); seen.update(claimed)
        values = list(unique.values()); rng.shuffle(values)
        n_val = max(1, int(len(values) * validation_fraction))
        if len(values) - n_val < 1:
            raise ValueError(f"too few distinct training examples for {dataset}/{cls}")
        pools["validation"][cls], pools["train"][cls] = values[:n_val], values[n_val:]
    for cls in classes:
        values, claimed = {}, {}
        for text in test[cls]:
            add(text, values, claimed)
        seen.update(values); seen.update(claimed)
        pools["test"][cls] = list(values.values())
    rows = []
    for cls in classes:
        for split in SPLITS:
            positives = pools[split][cls][:]
            # Equal allocation from other classes, followed by balanced sampling.
            others = [pools[split][c][:] for c in classes if c != cls]
            for pool in others:
                rng.shuffle(pool)
            per_class = (len(positives) + len(others) - 1) // len(others)
            negatives = [text for pool in others for text in pool[:per_class]]
            rng.shuffle(negatives); rng.shuffle(positives)
            n = min(len(positives), len(negatives))
            if n < 1:
                raise ValueError(f"no balanced examples remain for {dataset}/{cls}/{split}")
            for label, texts in [(1, positives[:n]), (0, negatives[:n])]:
                for text in texts:
                    key = text_id(text)
                    rows.append({"id": key, "group": key, "task": f"{dataset}/{cls}",
                                 "dataset": dataset, "split": split, "label": label, "text": text})
    return rows


def prepare(args):
    # Optional upstream loader, not the upstream evaluation/training loop.
    from sae_bench.sae_bench_utils.dataset_info import chosen_classes_per_dataset
    from sae_bench.sae_bench_utils.dataset_utils import get_multi_label_train_test_data
    import sae_bench
    if args.train_size < 8 or args.test_size < 4 or not 0 < args.validation_fraction < 1:
        raise ValueError("invalid dataset sample sizes or validation fraction")
    output = Path(args.output)
    if output.exists():
        raise ValueError("task file already exists; choose a new output")
    dedup_key = text_id
    if args.tokenizer:
        # Also drop texts that only differ after the collector's truncation, which
        # collect() would otherwise refuse as cross-split leakage.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
        dedup_key = truncated_token_key(tokenizer, args.context_length)
    rows = []
    for dataset in args.datasets:
        print(f"Preparing {dataset}", flush=True)
        train, test = get_multi_label_train_test_data(dataset, args.train_size, args.test_size, args.seed)
        rows.extend(binary_rows(dataset, train, test, chosen_classes_per_dataset[dataset], args.seed,
                                args.validation_fraction, dedup_key))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    read_tasks(partial)
    partial.replace(output)
    try:
        commit = subprocess.check_output(["git", "-C", str(Path(sae_bench.__file__).parent), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write_json(output.with_suffix(".provenance.json"), {"arguments": vars(args), "sha256": file_hash(output),
        "saebench_commit": commit, "tasks": len({r['task'] for r in rows}),
        "note": "Source train is split into train/validation after deduplication; not exact SAEBench score reproduction."})


def checkpoint_info(path):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if state.get("format") != FORMAT:
        raise ValueError("expected stage-2 Top-K checkpoint")
    source = state["data_manifest"]
    identity = {k: source[k] for k in ["model", "resolved_model_revision", "layer", "hook_point", "d_in", "burn_in_excluded"]}
    if identity["hook_point"] != f"block_output:{identity['layer']}":
        raise ValueError("probe collector currently supports block_output:<zero-based-layer> only")
    return identity


def blocks(model):
    for parent_name, child in [(None, "layers"), ("gpt_neox", "layers"), ("model", "layers"), ("transformer", "h")]:
        parent = model if parent_name is None else getattr(model, parent_name, None)
        result = getattr(parent, child, None) if parent is not None else None
        if result is not None:
            return result
    raise ValueError("cannot locate transformer block stack")


def check_token_leakage(rows, encoded):
    registry = {}
    for row in rows:
        token_key = tuple(encoded[text_id(row['text'])])
        key = (row['task'], token_key)
        if key in registry and registry[key] != row['split']:
            raise ValueError("identical truncated token sequences cross probe splits; deduplicate/rebuild task data")
        registry[key] = row['split']


@torch.no_grad()
def collect(args):
    from transformers import AutoModel, AutoTokenizer
    rows = read_tasks(args.tasks)
    identity = checkpoint_info(args.checkpoint)
    if args.model and args.model != identity['model']:
        raise ValueError("collector model must match the checkpoint model")
    if args.revision and args.revision != identity['resolved_model_revision']:
        raise ValueError("collector revision must match the checkpoint revision")
    if args.batch_size < 1 or args.context_length <= identity['burn_in_excluded']:
        raise ValueError("invalid batch size or context length")
    output = Path(args.output)
    signature = {"tasks_sha256": file_hash(args.tasks), "identity": identity,
        "model": args.model or identity['model'], "revision": args.revision or identity['resolved_model_revision'],
        "context_length": args.context_length, "dtype": args.dtype, "batch_size": args.batch_size,
        "pooling": "mean SAE features over non-special, non-padding tokens after stored burn-in"}
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        saved = json.loads((output / 'manifest.json').read_text())
        if saved['signature'] != signature:
            raise ValueError("activation cache settings/data changed")
        for chunk in saved['chunks']:
            if file_hash(output / chunk['file']) != chunk['sha256']:
                raise ValueError("activation cache checksum mismatch")
        print("Reusing verified activation cache", flush=True)
        return
    if any(output.iterdir()):
        raise ValueError("incomplete activation cache: choose a new output directory")
    tokenizer = AutoTokenizer.from_pretrained(signature['model'], revision=signature['revision'])
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    texts = {}
    for row in rows:
        texts.setdefault(text_id(row['text']), row['text'])
    ids = sorted(texts)
    # Use no automatic BOS/EOS, matching LeJEPA-SAE extraction.
    encoded = {key: tokenizer(texts[key], add_special_tokens=False, truncation=True,
                   max_length=args.context_length)['input_ids'] for key in ids}
    if any(not values for values in encoded.values()):
        raise ValueError("empty tokenized example")
    check_token_leakage(rows, encoded)
    model = AutoModel.from_pretrained(signature['model'], revision=signature['revision'],
                   torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    stack = blocks(model)
    layer = int(identity['layer'])
    if not 0 <= layer < len(stack):
        raise ValueError("checkpoint layer out of range")
    captured = []
    def hook(_module, _inputs, value):
        captured.append((value[0] if isinstance(value, tuple) else value).detach())
    handle = stack[layer].register_forward_hook(hook)
    chunks = []
    try:
        for start in range(0, len(ids), args.batch_size):
            chosen = ids[start:start + args.batch_size]
            tokens = tokenizer.pad({'input_ids': [encoded[k] for k in chosen]}, padding=True, return_tensors='pt')
            tokens = {k: v.to(args.device) for k, v in tokens.items()}
            captured.clear()
            model(**tokens, use_cache=False)
            if len(captured) != 1 or captured[0].shape[-1] != identity['d_in']:
                raise ValueError("wrong activation hook or dimension")
            mask = tokens['attention_mask'].bool()
            mask[:, :identity['burn_in_excluded']] = False
            for special in tokenizer.all_special_ids:
                mask &= tokens['input_ids'] != special
            if not mask.any(1).all():
                raise ValueError("example has no tokens after masking; clean task data before collection")
            # No token compaction before encoding: mask is retained explicitly.
            file = f"batch-{len(chunks):06d}.pt"
            torch.save({'ids': chosen, 'h': captured[0].cpu(), 'mask': mask.cpu()}, output / file)
            chunks.append({'file': file, 'sha256': file_hash(output / file), 'count': len(chosen)})
            if len(chunks) % 100 == 0:
                print(f"Collected {start + len(chosen)}/{len(ids)} texts", flush=True)
    finally:
        handle.remove()
    write_json(output / 'manifest.json', {'format': 'sae-jepa-probe-activations-v1', 'signature': signature,
        'ids': ids, 'chunks': chunks, 'resolved_model_commit': getattr(model.config, '_commit_hash', None),
        'tokenizer_vocab_sha256': hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()})
