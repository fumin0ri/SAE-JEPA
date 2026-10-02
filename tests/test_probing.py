import json
from pathlib import Path

import pytest
import torch

from conftest import tiny_config
from sae_jepa.probe_data import (SPLITS, binary_rows, checkpoint_info, check_token_leakage,
                                file_hash, read_tasks, text_id)
from sae_jepa.probing import aggregate, fit_task, main, select_features
from sae_jepa.stage2 import Stage2Config, Stage2Trainer
from sae_jepa.train import Trainer


def examples():
    return [{'id': f'{split}-{i}', 'group': f'{split}-{i}', 'task': 'task', 'dataset': 'synthetic',
             'text': f'{split} example {i}', 'split': split, 'label': i % 2}
            for split in SPLITS for i in range(12)]


def test_test_labels_cannot_change_selection_or_fit():
    rows = examples()
    g = torch.Generator().manual_seed(42)
    features = {text_id(r['text']): torch.tensor([r['label'] * 3., float(torch.randn((), generator=g)), 0.]) for r in rows}
    a = fit_task(features, rows, [1, 2], [.1, 1.], 100, include_test=True)
    flipped = [{**r, 'label': 1-r['label']} if r['split'] == 'test' else r for r in rows]
    b = fit_task(features, flipped, [1, 2], [.1, 1.], 100, include_test=True)
    assert a['1']['feature_indices'] == [0]
    assert a['1']['test']['accuracy'] == 1. and b['1']['test']['accuracy'] == 0.
    for k in ['1', '2']:
        assert {key: v for key,v in a[k].items() if key != 'test'} == {key: v for key,v in b[k].items() if key != 'test'}
    assert 'test' not in fit_task(features, rows, [1], [1.], 50)['1']


def test_selection_and_constant_features():
    x = torch.zeros(20, 4); y = torch.arange(20) % 2
    assert select_features(x, y, 2).tolist() == [0, 1]
    with pytest.raises(ValueError):
        select_features(x, torch.zeros(20), 1)


def test_split_leakage_and_data_preparation(tmp_path):
    rows = examples(); path = tmp_path/'tasks.jsonl'
    def save():
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    save(); assert len(read_tasks(path)) == 36
    rows[12]['group'] = rows[0]['group']; save()
    with pytest.raises(ValueError, match='crosses'):
        read_tasks(path)
    rows = examples(); rows[12]['text'] = rows[0]['text']; save()
    with pytest.raises(ValueError, match='crosses'):
        read_tasks(path)
    rows = examples()
    with pytest.raises(ValueError, match='truncated'):
        check_token_leakage(rows, {text_id(r['text']): [1,2] for r in rows})
    train = {c:[f'{c} train {i}' for i in range(20)] for c in ['a','b']}
    test = {c:[f'{c} test {i}' for i in range(10)] for c in ['a','b']}
    out = binary_rows('data',train,test,['a','b'],42,.2)
    assert out == binary_rows('data',train,test,['a','b'],42,.2)
    rows = out; save(); read_tasks(path)


def test_macro_aggregation_distinguishes_datasets():
    tasks = {str(i): {'dataset': 'a' if i < 3 else 'b', 'scores': {'1': {'test': {'accuracy': 1. if i < 3 else 0.}}}} for i in range(4)}
    r = aggregate(tasks, 'test', 1)
    assert r['dataset_macro_accuracy'] == .5 and r['task_macro_accuracy'] == .75


@pytest.mark.parametrize('frontend_kind', ['dense', 'raw', 'zca'])
def test_offline_checkpoint_probe_end_to_end(manifest, tmp_path, frontend_kind):
    t = Trainer(tiny_config(manifest, tmp_path/'front', steps=1)); front = t.save_checkpoint()
    if frontend_kind != 'dense':
        from sae_jepa.stage2 import main as stage2_main
        stage2_main(['prepare-baselines', '--reference-checkpoint', str(front),
                     '--output', str(tmp_path/'baselines'), '--kinds', frontend_kind,
                     '--maximum-positions', '64'])
        front = tmp_path/'baselines'/f'{frontend_kind}.pt'
    cfg = Stage2Config(dictionary_size=16,k=4,steps=1,batch_size=16,amp_dtype='none',
                       calibration_batches=1,eval_batch_size=16,eval_batches=1)
    trainer = Stage2Trainer(front, tmp_path/'stage2', cfg, 'cpu')
    checkpoint = trainer.save()
    # Synthetic manifest hook point is not a production block name. Make explicit
    # identity metadata for this mock cached activation test.
    state = torch.load(checkpoint, weights_only=False)
    state['data_manifest']['hook_point'] = f"block_output:{state['data_manifest']['layer']}"
    torch.save(state, checkpoint)
    rows = examples(); tasks = tmp_path/'tasks.jsonl'
    tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cache = tmp_path/'cache'; cache.mkdir()
    ids = [text_id(r['text']) for r in rows]
    g = torch.Generator().manual_seed(19)
    torch.save({'ids': ids, 'h': torch.randn(len(ids),3,16,generator=g),
                'mask': torch.tensor([[False,True,True]]*len(ids))},cache/'batch.pt')
    signature = {'tasks_sha256':file_hash(tasks),'identity':checkpoint_info(checkpoint)}
    (cache/'manifest.json').write_text(json.dumps({'format':'sae-jepa-probe-activations-v1',
        'signature': signature,'ids':ids,'chunks':[{'file':'batch.pt','sha256':file_hash(cache/'batch.pt')}] }))
    args = ['evaluate','--tasks',str(tasks),'--activations',str(cache),'--checkpoints',str(checkpoint),
            '--ks','1','2','--cs','1','--max-iter','30','--device','cpu']
    before = {k:v.clone() for k,v in state['sae'].items()}
    for name in ['one','two']:
        main(args+['--output',str(tmp_path/name)])
    a=json.loads((tmp_path/'one/model-00.json').read_text())
    b=json.loads((tmp_path/'two/model-00.json').read_text())
    assert a == b
    assert 'test' not in a['aggregate'] and (tmp_path/'one/summary.md').exists()
    if frontend_kind != 'dense':
        assert a['frontend_type'] == frontend_kind
        assert a['frontend_name'] in (tmp_path/'one/summary.md').read_text()
    after = torch.load(checkpoint,weights_only=False)['sae']
    assert all(torch.equal(before[k],after[k]) for k in before)


def test_collector_block_hook_mask_and_cache_reuse(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    import sae_jepa.probe_data as module
    rows=examples(); tasks=tmp_path/'tasks.jsonl'
    tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    identity={'model':'fake','resolved_model_revision':'rev','layer':0,
              'hook_point':'block_output:0','d_in':4,'burn_in_excluded':1}
    monkeypatch.setattr(module,'checkpoint_info',lambda _: identity)
    class Tokenizer:
        pad_token_id=0
        eos_token_id=0
        all_special_ids=[0]
        def __call__(self,text,**kw):
            return {'input_ids':[ord(c) for c in text][:kw['max_length']]}
        def pad(self,values,**kw):
            ids=values['input_ids']; n=max(map(len,ids))
            return {'input_ids':torch.tensor([v+[0]*(n-len(v)) for v in ids]),
                    'attention_mask':torch.tensor([[1]*len(v)+[0]*(n-len(v)) for v in ids])}
        def get_vocab(self): return {'pad':0}
    class Block(torch.nn.Module):
        def forward(self,x): return (x+1,)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layers=torch.nn.ModuleList([Block(),Block()]); self.config=SimpleNamespace(_commit_hash='rev')
        def forward(self,input_ids,**kw):
            x=input_ids.float().unsqueeze(-1).expand(-1,-1,4)
            for layer in self.layers: x=layer(x)[0]
            return x
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer()),
        AutoModel=SimpleNamespace(from_pretrained=lambda *a,**kw:Model())))
    args=SimpleNamespace(tasks=str(tasks),checkpoint='unused',output=str(tmp_path/'cache'),
        batch_size=8,context_length=64,dtype='float32',device='cpu',model=None,revision=None)
    module.collect(args)
    manifest=json.loads((tmp_path/'cache/manifest.json').read_text())
    batch=torch.load(tmp_path/'cache'/manifest['chunks'][0]['file'],weights_only=True)
    row=next(r for r in rows if text_id(r['text'])==batch['ids'][0])
    assert batch['h'][0,0,0] == ord(row['text'][0])+1
    assert not batch['mask'][:,0].any()
    assert int(batch['mask'][0].sum()) == len(row['text'])-1
    module.collect(args)
