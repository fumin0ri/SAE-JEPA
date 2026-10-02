# A/B baseline：Raw / ZCA → Top-K SAE

`sj-stage2 prepare-baselines`で固定前段を作り、既存の`train` / `sweep` /
`evaluate` / `sj-probe`をそのまま使います。前段のニューラルネット学習はありません。

| 条件 | SAEに渡す表現（Stage2 calibration前） | 元の活性への復元 |
|---|---|---|
| A：Raw | `x = (h − μ) / s` | `h = s x + μ` |
| B：ZCA whitening | `y = (x − c) W` | `h = s (y W⁻¹ + c) + μ` |

`μ, s`は参照checkpointのtrain正規化統計を共有します。Bはtrainで推定した
`c = mean(x)`、`C = Cov(x)`から`W = U diag((eigenvalues + ε)^−1/2) Uᵀ`を作ります。
全次元を保持し、再構成には保存した変換の逆行列を使います。ridge readoutは不要です。
εによる正則化があるため、変換後の共分散は厳密な単位行列とは限りません。
このBはZCAであり、別コマンド`sj-fit-pca`が作るPCA座標の白色化とは区別します。

## 1. 比較対象に合わせた前段の作成

以下はリポジトリルートで実行するBash例です。`REFERENCE`には、比較したいCの
**Stage1 checkpoint**を指定します。masked encoderならreadout付きcheckpointです。
参照から使うのはデータの同一性・split・burn-in/先頭除外・正規化で、encoderや
readoutの重みはA/Bに使いません。参照の入力次元と潜在次元は同じである必要があります。

```bash
pip install -e '.[dev,probe]'
REFERENCE=/path/to/beta10-readout.pt
BASELINES=runs/baseline-frontends

sj-stage2 prepare-baselines \
  --reference-checkpoint "$REFERENCE" \
  --output "$BASELINES" \
  --maximum-positions 262144 --sample-seed 1729 \
  --epsilon 0.0001 --chunk-size 2048 --device cpu
```

- `raw.pt` / `zca.pt`：Stage2用固定前段。`raw.json` / `zca.json`に設定・hash・推定履歴。
- `--kinds raw`でAだけ、`--kinds zca`でBだけを作れます。既定は両方です。
- 既定はtrain全体から262,144トークンを固定seedで非復元抽出します。trainがそれより
  少なければ全件です。validation/testの値は読みません。
- `--maximum-positions 0`は全trainを使います。この場合はshardをメモリに読み込むので、
  大規模データでは既定の上限付き抽出を推奨します。
- ZCA推定はfloat64、保存と適用はfloat32です。変換・逆変換にはbf16 autocastを適用しません。
  4096次元では共分散・固有値分解に相応のメモリと時間を使います。`--device cuda`も指定可能です。
- 活性を移動した場合は`--activation-manifest /new/path/manifest.json`を追加します。
  元データのfingerprint・split・除外設定は引き続き検証します。
- 出力が空でなければ停止します。白色化の設定を変えるときは別ディレクトリを使います。

## 2. A/BのStage2学習

```bash
sj-stage2 sweep \
  --checkpoints "$BASELINES/raw.pt" "$BASELINES/zca.pt" \
  --config configs/stage2_topk.yaml \
  --output runs/stage2-baselines --device cuda
```

`model-00`がA、`model-01`がBです。途中再開は同じコマンドに`--resume`を追加します。
この設定は辞書16,384、K=64、batch512、10k updates、seed42です。
共通のtrain-only calibration（平均＋全座標共通スカラー）をA/Bにも適用し、
初期SAE重み・データ順・学習予算・評価サンプルを揃えます。

Cも一度に比較する場合は、`--checkpoints`の最後に`"$REFERENCE"`を追加して
別の出力先で実行します。既存CのStage2結果を再利用するときは、そのrunの
`resolved_config.json`を`--config`へ渡し、**全設定と学習stepを一致**させてください。
比較用のprobeは設定・step・初期SAE hashが一致しないcheckpointを拒否します。

レポートは`report/validation.md`・`.json`・`.png`に保存します。
Raw/ZCAを名前で区別し、元の活性空間のFVU、latent FVU、実測L0、評価時非発火率を併記します。
前段単体のFVUは数値誤差を除きゼロです。Bのlatent FVUとAのlatent FVUは異なる距離尺度なので、
前段間の再構成比較には元の活性空間のend-to-end FVUを使ってください。

単独評価はStage2 checkpointに前段を同梱しているため、元の`raw.pt`/`zca.pt`なしで実行できます。

```bash
sj-stage2 evaluate \
  --checkpoint runs/stage2-baselines/model-00/checkpoints/latest.pt \
  --output runs/stage2-baselines/raw-validation.json --device cuda
```

## 3. 同一キャッシュでsparse probe

既存のAG News活性キャッシュとタスクJSONLを共有できます。同じLLM・layer・除外条件かは
`sj-probe`が検証します。キャッシュがなければ[probing手順](probing.md)の`collect`を先に実行します。

```bash
sj-probe evaluate \
  --tasks data/probe/ag-news.jsonl \
  --activations data/probe/ag-news-activations \
  --checkpoints \
    runs/stage2-baselines/model-00/checkpoints/latest.pt \
    runs/stage2-baselines/model-01/checkpoints/latest.pt \
  --output runs/probe-baselines --device cuda
```

Cを加える場合は最後にCのStage2 checkpointを追加します。主指標はTop-1 accuracy、
補助はTop-2/5と元の活性空間のFVUです。まずvalidationで条件を固定し、最終確認では
別の出力先で`--include-test`を指定します。seedの追加は`--set seed=43`などで別runとして行い、
前段推定seedとSAE学習seedを区別してください。

10k updatesは同予算のpilotです。前段の推定・学習コストは別途扱い、収束後の優劣とは区別します。
