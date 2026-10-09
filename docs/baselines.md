# A/B baseline：Raw / ZCA → Top-K SAE

## 論文を参考にしたRaw / PCA、100k step

[Data Whitening Improves Sparse Autoencoder Learning, §5.2](https://arxiv.org/html/2511.13981v1#S5.SS2)
のPCA変換と逆変換後の再構成損失に対応する場合は、次を使用します。
既存のZCA条件とは別条件です。

```bash
pip install -e .
REFERENCE=runs/stage1/ablation-lambda-0-beta-0-dim-256-100k-decay50/checkpoints/latest.pt \
ACTIVATION_MANIFEST=/home/iwasaki/LeJEPA-SAE/data/the-pile/pythia-6.9b/layer-16-ctx1024-100m/manifest.json \
SEEDS="42 43 44" \
bash scripts/baseline_sae_100k.sh
```

`REFERENCE`は手元にある入力次元＝出力次元のStage1 checkpointへ変更してください。
利用するのはtrain正規化・split・先頭除外の設定だけで、encoder重みは使いません。
Rawにも平均除去と共通スカラー正規化、両条件にStage2 calibrationが入ります。
正規化も一切しない生の数値入力を意味するものではありません。

PCAはtrainから262,144位置を抽出し、`x=(h-mu)/s`の標本共分散（分母n−1）を推定します。
行ベクトル表記で `z=(x-c) U diag((lambda+epsilon)^(-1/2))`、全次元保持、
固有値の降順です。保存した変換の逆行列で元activationに戻します。
`epsilon=1e-4`はこのリポジトリでの選択（スカラー正規化後の単位）であり、
論文本文に指定された値ではありません。fit件数も今回の実験設定です。

`reconstruction_space=activation`では元のactivationのMSEに勾配を流します。
既定の`latent`は従来どおりcalibration後のSAE入力空間のMSEです。
論文のモデル・K・学習予算すべてを再現する実験ではありません。
辞書65,536、K=64、batch512、100,000 step、lr=1e-4、warmup500、末尾20% decay。
1条件・1seedあたり51.2M提示トークン。既存のlatent損失の実験とは損失も異なります。

出力は `runs/baseline-raw-pca-64k-100k/seed-*/model-00` がRaw、`model-01`がPCA。
1seedだけなら`SEEDS=42`。再開は同じ環境変数に`RESUME=1`を追加します。
`RUN_ROOT`で保存先を変更できます。前段fit途中の失敗時は新しい出力先を使ってください。
比較には逆変換後の`end_to_end.fvu`を使います。

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
