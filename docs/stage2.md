# 第2段階：固定したDense前段とTop-K SAEの比較

既存のλ=0 / 0.0003 / 0.001のcheckpointを使う小規模比較です。
第1段階の再学習と全データの正規化再計算は不要です。

## 実行

```bash
pip install -e .
STAGE1_ROOT=runs/stage1-skip-lead-100k-v2 \
RUN_ROOT=runs/stage2-pilot \
WEIGHTS="0 0.0003 0.001" \
STEPS=10000 SEED=42 DICTIONARY_SIZE=16384 K=64 \
bash scripts/stage2_sweep.sh
```

`STAGE1_ROOT`は意図したdecay_fraction=0.5のcheckpoint保存先を指定してください。
同じディレクトリを別実験で上書きした場合、古いcheckpointは復元されません。
各runの`provenance.json`に前段の設定・step・重みハッシュを保存します。

前段checkpointは既定で`lambda-*/seed-42/checkpoints/latest.pt`から読みます。
前段seedの指定は`STAGE1_SEED`、SAEの初期化と学習データ順は`SEED`です。
`ACTIVATION_MANIFEST`で活性manifestの移動先、`DEVICE=cpu`でCPUを指定できます。

既定は各モデル10k updates、batch 512、辞書16,384、K=64です。
学習トークン提示回数は各条件5,120,000（再利用があれば重複を含む）。
validationで比較し、条件選択中はtestを既定で使用しません。
まず短く動作確認する場合：

```bash
STAGE1_ROOT=runs/stage1-skip-lead-100k-v2 \
RUN_ROOT=runs/stage2-smoke STEPS=100 \
bash scripts/stage2_sweep.sh \
  --set warmup_steps=10 --set calibration_batches=4 \
  --set eval_batches=4 --set validation_every=0
```

追加の`--set`は`configs/stage2_topk.yaml`の**フラットなキー**を指定します。
例：`--set lr=0.0002`、`--set batch_size=256`。
動作確認用の少量評価を、本実験の性能比較に使わないでください。

## 学習対象と比較条件

```text
h → 固定encoder → y → (y−μy)/sy → Top-K SAE → û
                                        ↓
                     ŷ = sy û + μy → 固定decoder → ĥ
```

- `μy, sy`は**trainのみ**から64バッチをサンプリングして推定し、その後固定します。
  全条件でcalibration seed・データ順を共有します。`sy`は全座標共通のスカラーです。
  λによる平均・全体振幅の差を除き、共分散の異方性を保持します。白色化やトークンごとの正規化はしません。
- 損失はSAE入力空間のMSE `mean((û−u)^2)`。第1段階へ勾配は流しません。
- SAEは各トークンで上位K個を選択してReLUを適用します。正値が少なければ実際の発火数はK未満です。
- encoder/decoderは学習中は非共有。初期encoderはdecoderの転置、decoder列は単位ノルム。
  decoder勾配の半径方向成分を除き、更新後も列を単位ノルムへ戻します。
- AdamW、warmupと末尾linear decayを使用します。AuxK、dead feature再初期化、L1損失は使わないplain Top-Kのpilotです。
- 全候補を学習開始前に確認し、活性manifest、split、先頭除外、入力正規化、モデル寸法の違いを拒否します。
  学習開始前のSAEパラメータのハッシュ、サンプルハッシュ、前段重みハッシュを保存します。
- 前段学習時の先頭除外・burn-in・splitを引き継ぎます。全λで学習順、SAE初期化seed、辞書サイズ、K、update数を共有します。
- testを最終確認で使う場合、学習済みstage-2 checkpointを`evaluate --split test`で評価できます。

## 出力と読むべき指標

`RUN_ROOT/report/validation.md`、`.json`、`.png`が比較レポートです。
`model-00`以降は渡したcheckpointの順番です。λは各JSONにも記録されます。

| 指標 | 意味 |
|---|---|
| `frontend.fvu` | 前段単体 `h→y→h_front` の元の活性空間FVU |
| `end_to_end.fvu` | **主指標**。前段＋SAE＋固定decoderで得た`ĥ`の元の活性空間FVU |
| `extra_fvu` | 最終FVU−前段単体FVU。誤差の直交分解ではなく、差は負にもなり得る |
| `latent.fvu` | `y`に対する`ŷ`のFVU。これだけで前段間の優劣を決めない |
| `l0_mean/min/max` | 正値のSAE特徴数（K以下） |
| `inactive_fraction` | 今回の評価サンプル中に一度も発火しなかった特徴割合 |
| `train_never_fired_fraction` | これまでの学習全体で一度も発火しなかった割合 |
| `feature_firing_counts` | 各特徴の評価サンプル内の発火回数 |

FVU分母は評価した全トークンの平均で中心化した分散です。
上記の非発火率は永続的なdead featureの証明ではなく、観測窓を明記した診断です。
同じKでも平均L0が違う場合は、再構成だけでなく実際の疎性も考慮してください。
pilotでは収束や複数seedでの優位性までは保証しません。

各runに`resolved_config.json`、`provenance.json`、`metrics.jsonl`、
`eval-validation.json`、`checkpoints/latest.pt`を保存します。
前段の重み・正規化、潜在calibration、SAE、optimizer/scheduler、データiterator、発火累積をcheckpointに同梱します。
オフライン評価に元のstage-1 checkpointは不要ですが、活性データは必要です。

## 中断再開と単独評価

```bash
# 最初と同じ設定で再実行。完了済みrunは追加学習せず評価を再生成します。
STAGE1_ROOT=runs/stage1-skip-lead-100k-v2 \
RUN_ROOT=runs/stage2-pilot RESUME=1 \
bash scripts/stage2_sweep.sh

sj-stage2 evaluate \
  --checkpoint runs/stage2-pilot/model-01/checkpoints/latest.pt \
  --split test --output runs/stage2-pilot/model-01/eval-test.json --device cuda

# 全条件のtest評価後
sj-stage2 report --run-root runs/stage2-pilot --split test
```

resumeでは設定変更（総step数を含む）や前段重み変更を拒否します。
別の総step数を試す場合は新しい出力先で開始してください。
既存のrunへ新規学習を上書きしません。checkpointは原子的にlatest.ptを置換します。
再開は元の総学習計画を保ち、CPUで連続実行との一致をテストしています。
SAE学習にdropoutやランダム射影はなく、データ乱数は専用iteratorから復元します。
CUDA/bf16の再現性と実機メモリは環境ごとに確認してください。

単独学習は`sj-stage2 train --checkpoints PATH --output DIR --config configs/stage2_topk.yaml`。
Python直接実行は`python -m sae_jepa.stage2 ...`でも同じです。
