"""Compare a residual AE to its exact initial whitening on fixed validation data."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .config import config_from_dict
from .data import write_json
from .evaluate import evaluate_model, load_checkpoint_model, write_evaluation
from .models import WhitenedResidualAE
from .stage2 import source_for, tensor_hash


def diagnose(checkpoint, output, *, manifest=None, device='cuda', batches=64,
             amp_dtype='none'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be empty; choose a new directory')
    device = torch.device(device)
    model, state = load_checkpoint_model(checkpoint, device)
    if not isinstance(model, WhitenedResidualAE):
        raise ValueError('requires a whitened_residual_ae checkpoint')
    cfg = config_from_dict(state['config'])
    source = source_for(state, manifest)
    kwargs = dict(split='validation', batch_size=cfg.eval.batch_size,
                  maximum_batches=batches, device=device, amp_dtype=amp_dtype,
                  sigreg_cfg=cfg.sigreg, eval_cfg=cfg.eval, detailed=True)
    trained_hash = tensor_hash(model.state_dict())
    after = evaluate_model(model, source, **kwargs)
    # The first residual layer need not be reconstructed: the zero last layer
    # makes y=z for every input. Restore the decoder to the saved W inverse.
    model.initialize_identity()
    before = evaluate_model(model, source, **kwargs)
    if before['sample_sha256'] != after['sample_sha256']:
        raise ValueError('paired evaluation samples differ')
    keys = ['gaussian/heldout_sigreg', 'gaussian/diagnostic_sigreg',
            'gaussian/diagnostic_w2_sq', 'gaussian/cov_fro_dev',
            'gaussian/cov_effective_rank', 'gaussian/cov_participation_ratio',
            'gaussian/variance_mean', 'reconstruction/fvu']
    result = dict(checkpoint=str(checkpoint), step=state['step'],
                  sigreg_weight=cfg.sigreg.weight, amp_dtype=amp_dtype,
                  trained_model_sha256=trained_hash,
                  sample_sha256=after['sample_sha256'], positions=after['positions'],
                  evaluation_config=asdict(cfg.eval), sigreg_config=asdict(cfg.sigreg),
                  maximum_batches=batches,
                  initial_state='reconstructed y=z and exact inverse of saved whitening',
                  metrics={k: dict(before=before[k], after=after[k],
                                   delta=after[k]-before[k]) for k in keys})
    output.mkdir(parents=True, exist_ok=True)
    write_evaluation(before, output / 'before.json')
    write_evaluation(after, output / 'after.json')
    write_json(output / 'comparison.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--activation-manifest')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batches', type=int, default=64)
    p.add_argument('--amp-dtype', choices=['none', 'bfloat16'], default='none')
    a = p.parse_args()
    if a.batches < 1:
        p.error('--batches must be positive')
    result = diagnose(a.checkpoint, a.output, manifest=a.activation_manifest,
                      device=a.device, batches=a.batches, amp_dtype=a.amp_dtype)
    print(result['metrics'])


if __name__ == '__main__':
    main()
