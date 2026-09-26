# SAE-JEPA

LLM残差ストリームに対し、**Top-K SAEの前段として「再構成＋SIGReg」で密なGaussian化表現を作る**ための実験コードです。
[LeJEPA-SAE](https://github.com/fumin0ri/LeJEPA-SAE)（`extract`、safetensors）と [JEPA-SAE](https://github.com/fumin0ri/JEPA-SAE)（`sr-extract-pile`）の活性抽出フォーマットをそのまま読み込みます。

本リポジトリは **第1段階**（密な前段表現の学習と評価）を実装しています。第2段階では、ここで学習したencoderと正規化統計を固定し、Raw・PCA whitening・Dense-AE・Dense-SIGReg-AEの各前段を同じTopK SAEへ接続します（共通インターフェース `encode_dense(h)`、`sae_jepa.frontends`）。

## モデル（`model.type = dense_sigreg_ae`）

```text
x     = (h - mu) / s              mu: train平均ベクトル, s: train全座標共通のスカラー標準偏差
y     = G_theta(x)                Linear 4096->4096 -> GELU -> Linear 4096->4096（出力活性化なし）
x_hat = D_phi(y)                  Linear 4096->4096（列ノルム制約なし）
h_hat = s * x_hat + mu

L = L_rec + lambda_sigreg * L_SIGReg(y),    L_rec = mean((x_hat - x)^2)
s^2 = E_train ||h - mu||^2 / 4096
```

- `h`: Pythia-6.9B layer 16 残差（4,096次元）。
- 平均と全体スケールのみ揃え、相関・分布形状の変換はencoderに任せます。PCA whiteningはモデル内に入れず、独立した比較条件（`sj-fit-pca`, `frontends.PCAWhiteningFrontend`）として用意しています。
- BatchNorm/LayerNorm、ReLU出力、TopK、L1、発火率損失はありません。
- `L_rec` は正規化空間でのMSEで、train尺度でのFVUに相当します。

## SIGReg（`sae_jepa/sigreg.py`）

LeJEPAのEpps–Pulley型：

```text
u_im = a_m^T y_i   （a_m は単位球面上一様、trainではstepごとに再サンプリング）
L_SIGReg = (N/M) sum_m ∫ | (1/N) sum_i exp(i t u_im) - exp(-t^2/2) |^2 exp(-t^2/2) dt
```

| 項目 | 既定値 |
|---|---|
| 射影数 M | 256 |
| 積分 | [-5, 5] の17点、台形則 |
| 計算 | cos/sinで実部・虚部、float32（autocast無効） |
| N係数 | あり（`sigreg.scale_by_batch_size`） |
| train射影 | 専用generator（`seed` と `sigreg.seed_offset` から生成）、stepごとに再サンプル |
| validation射影 | 専用seedの固定射影。trainの乱数状態は変更しない |

厳密に N(0, I) なサンプルでも有限標本誤差により N係数付き統計量は射影あたり
`sum_j w_j (1 - phi_j^2) phi_j ≈ 1.06` となります（`sigreg.gaussian_expected_value`）。この値がノイズフロアです。
損失の規約は各runの `sigreg_convention.json` とcheckpointに記録されます。旧RDMの重み（3・6・12・24）とは数値の意味が異なります。

## インストール

```bash
pip install -e ".[dev]"
```

## データ

`ACTIVATION_MANIFEST`（`data.activation_manifest`）には、次のどちらの `manifest.json` も指定できます。形式は自動判定します。

| 抽出元 | manifest | shard | 使う位置 |
|---|---|---|---|
| LeJEPA-SAE `extract` | `format_version: 1`、`d_llm`、`shards[].split / sequences[].offset,length` | `*.safetensors`（`activations: [num_tokens, d]`） | 全トークン（burn-inなし） |
| JEPA-SAE `sr-extract-pile` | `format: shared-residual-sequence-shards-v2` | `*.pt`（`activations: [n, T, d]`, `valid_lengths`） | `[burn_in_tokens, valid_length)`（`data.skip_burn_in`） |

LeJEPA-SAEのshardはmemory mapで必要な行だけ読みます（`safetensors` パッケージは不要）。
LeJEPA-SAEのmanifestは文書単位のtrain / validation / testを持つので、そのsplitをそのまま使います。

**先頭トークンの除外（`data.skip_leading_positions=k`、sweepでは `SKIP_LEADING_POSITIONS=k`）**：保存された各系列の位置 `< k` を、正規化統計・学習・評価のすべてから除きます。位置0は各forward（LeJEPA-SAEでは `context_length` ごとの区切り）の先頭トークンです。Pythia-6.9B layer 16ではこの位置の ‖x‖²/d が他の約300倍あり、0.13%のサンプルで入力分散の約30%を占めます。そのため `s` とFVUの分母が歪みます。既定は0（除外しない＝従来と同じ挙動）です。JEPA-SAE形式ではmanifestのburn-inと合わせて `max(burn_in, k)` 未満の位置を除きます。
- 正規化統計にも除外した位置を記録し、runの設定と一致しなければ学習前にエラーにします。`sj-compute-normalization --skip-leading-positions k` で作り直してください。
- 第2段階では `DenseCheckpointFrontend.skip_leading_positions` を見て、同じ位置を除いてください。

- train: manifestの `train` shard。**正規化統計はtrainからのみ計算**します。
- validation / test: manifestに `test` があればそれを（LeJEPA-SAE形式は通常こちら）、なければ `validation` shardの後半半分をtestとして予約します（`data.test_split=auto`, `holdout_test_fraction=0.5`）。validation shardが1個しかない場合は、そのshard内で**系列単位**に分割します（`path#rows=start:stop`）。分割できない場合（系列が1本など）は学習前にエラーで停止します。shard・系列範囲の重複もエラーになります。
- 学習開始時に `eval.required_splits`（既定 `[validation, test]`、sweepでは `EVAL_SPLITS` から設定）の各splitが空でなく、評価バッチを1つ以上作れることを確認します。validationだけで評価する場合は `data.test_split=none` と `eval.required_splits=[validation]`（sweepでは `EVAL_SPLITS=validation`）を指定してください。

## 実行

### Pilot sweep（Dense-AE と Dense-SIGReg-AE）

```bash
ACTIVATION_MANIFEST=../LeJEPA-SAE/data/the-pile/pythia-6.9b/layer-16-ctx1024-100m/manifest.json \
RUN_ROOT=runs/stage1-pilot \
WEIGHTS="0 0.01 0.1 1" SEED=42 STEPS=10000 \
bash scripts/stage1_sweep.sh
```

1. train-onlyの `mu, s` を一度だけ計算（`normalization.pt`）し全条件で共有
2. 各λで学習（λ=0はDense-AE：SIGRegの計算も乱数消費も行わない）
3. validation / testで詳細評価
4. `RUN_ROOT/report/` に `{split}.csv|json|md` と `{split}_tradeoff.png`（FVU vs held-out SIGReg）

全条件でデータ順序（`train.seed`由来）とモデル初期値（`torch.manual_seed(train.seed)`）が一致し、SIGReg用乱数は別管理です。
50k stepへの延長は、新しい学習計画として `STEPS=50000` と別の `RUN_ROOT` で実行してください（schedulerが総step数に依存するため、10k runの再開では延長しません）。

先頭トークンを除外した比較は、正規化統計が変わるので別の `RUN_ROOT` で実行します：

```bash
ACTIVATION_MANIFEST=../LeJEPA-SAE/data/the-pile/pythia-6.9b/layer-16-ctx1024-100m/manifest.json \
RUN_ROOT=runs/stage1-skip-lead-50k SKIP_LEADING_POSITIONS=1 \
WEIGHTS="0 0.0003 0.001 0.003" SEED=42 STEPS=50000 \
bash scripts/stage1_sweep.sh
```

### 個別コマンド

```bash
sj-compute-normalization --activation-manifest MANIFEST --output runs/x/normalization.pt
sj-train-dense --config configs/dense_sigreg_ae.yaml \
  --set data.activation_manifest=MANIFEST \
  --set data.normalization_path=runs/x/normalization.pt \
  --set sigreg.weight=0.1 --set train.output_dir=runs/x/lambda-0p1
sj-evaluate-dense --checkpoint runs/x/lambda-0p1/checkpoints/latest.pt --split test
sj-report-dense --run-root runs/x
sj-fit-pca --activation-manifest MANIFEST --normalization runs/x/normalization.pt --output runs/x/pca.pt
```

`sj-train-dense` は `--resume auto`（既定）で `OUTPUT/checkpoints/latest.pt` から再開します。
再開時は、ログ間隔・checkpoint間隔・出力先・device・`eval.*` 以外のすべての設定（model / sigreg / optim / data / seed）をcheckpointと比較し、1つでも異なれば停止します（例：`optim.weight_decay`、`sigreg.scale_by_batch_size`）。split割り当てと正規化統計の一致も確認します。設定を変える場合は新しいrunとして学習してください。

### CPUスモークテスト

```bash
bash scripts/smoke.sh
```

## 既定の学習条件（pilot）

| 項目 | 値 |
|---|---|
| batch size | 512（勾配累積なし） |
| optimizer | AdamW, lr 1e-4, weight decay 0 |
| schedule | warmup 1,000 step → 一定 → 最後の20%で線形にゼロへ |
| steps | 10k |
| seed | 42 |
| SIGReg重み | 0, 0.01, 0.1, 1（探索の出発点であり最適値ではない） |
| AMP | bf16 autocast（SIGRegはfloat32） |

## ログと評価

学習中（`metrics.jsonl`）:

- 再構成損失、SIGReg、重み付きSIGReg、学習率
- **y に対する勾配RMS**：`grad_rms_y/reconstruction`、`grad_rms_y/sigreg_weighted` とその比
- 定期validation：残差空間のMSE・FVU、held-out SIGReg、平均のずれ、分散

checkpoint評価（`eval-{split}-step-*.json`）:

| 軸 | 指標 |
|---|---|
| 情報保持 | 元残差空間の MSE・FVU（FVUは評価split全体の平均を基準） |
| Gaussian化 | held-out SIGReg（固定validation射影）、`mean_sq_per_dim`、分散、`cov_fro_dev = ‖Cov(y)−I‖_F/√d`、固有値スペクトル・有効ランク |
| 診断 | 学習に使わない射影上の W₂²・SIGReg・分位点のずれ |
| 外れ値診断 | 下記 |

### 外れ値・低ランク化の診断

ランダム射影のSIGRegは、(a) ごく少数の極端なサンプルと、(b) 全分散を保ったまま低次元の部分空間に集中する低ランク化の、どちらもほとんど検出できません。詳細評価ではこの2つを切り分ける指標を出します。

| 指標 | 意味 |
|---|---|
| `gaussian/y_sqnorm_q{0.5,0.9,0.99,0.999,1}` | 各サンプルの ‖y‖²/d の分位点（Gaussianなら≈1±0.02、`reference/...` と比較） |
| `input/x_sqnorm_*`、`reconstruction/sample_error_*` | 正規化入力 ‖x‖²/d と再構成誤差のサンプル分布（外れ値が入力由来かを見る） |
| `leading/*`、`nonleading/*` | 系列内の位置が `eval.leading_positions` 未満（既定：先頭トークン）とそれ以外の平均 ‖y‖²/d・‖x‖²/d・誤差、SSEに占める割合 |
| `reconstruction/fvu_excluding_leading` | 先頭トークンを除いたFVU |
| `outliers/*` | ‖y‖²上位 `eval.outlier_fraction`（既定0.1%）の件数、先頭トークンの割合、‖y‖²総和に占める割合、系列内位置のヒストグラム |
| `outliers/top` | ‖y‖²上位 `eval.outlier_table_size` 件の shard・系列・位置・‖y‖²・‖x‖²・誤差（JSONのみ） |
| `gaussian/excl_leading/cov_*` | 先頭トークンを除いた共分散（‖Cov−I‖、最大固有値、有効ランクなど） |
| `gaussian/trimmed/cov_*` | ‖y‖²上位を除いた共分散（全体のモーメントから厳密に差し引き） |

- 外れ値が原因なら、`excl_leading` か `trimmed` で有効ランクと最大固有値が参照値に近づきます。
- 低ランク化が原因なら、除いても有効ランクは低いままです。
- 固有値スペクトル（全体・先頭除外・上位除外、同じ件数のGaussian参照）は `eval-*-spectra.pt` に保存し、レポートでは `{split}_spectra.png` に描きます。

既存のcheckpointも、学習し直さずにこの診断で再評価できます（新しい `eval.*` 設定には既定値が入ります）：

```bash
RUN_ROOT=runs/stage1-pilot bash scripts/evaluate_stage1.sh
```

診断の設定は `EVAL_ARGS="--set eval.leading_positions=4 --set eval.outlier_fraction=0.01"` のように変えられます（`eval.*` 以外は変更不可）。

**評価バッチ**：同じ文書内の連続トークンは強く相関するため、保存順のバッチでは各トークンの分布がGaussianでもバッチ単位のSIGRegが大きく出ます。そこで評価専用の固定seed（`eval.sample_seed`）で**split全体（全shard・全系列）から** `eval.batches × eval.batch_size` 個の位置を非復元抽出し、ランダムな順序でバッチに割り当てます（`eval.batches=0` なら全フルバッチ）。抽出はデータとseedだけで決まるため、全λ・全checkpointが同一のバッチで評価されます。読み込みはmemory mapで行い、必要な行だけを読みます。

すべてのGaussian化指標に、**同じバッチ数・バッチサイズ・射影数の厳密なGaussianサンプルでの参照値**（`reference/...`）を併記します。
dead featureや「正なら発火」などの疎モデル用指標は密モデルには適用しません。

レポートは (FVU, held-out SIGReg) のPareto frontを示すだけで、最低SIGRegのcheckpointを自動的に勝者にはしません。

## checkpoint内容

重み、optimizer、scheduler、乱数状態（torch/cuda/SIGReg generator/データiterator）、`mu, s`、データmanifest記録（fingerprint・split別shard）、解決済みconfig、SIGReg規約。

既定では `train.checkpoint_every` stepごとに `checkpoints/latest.pt` だけを上書き保存します（4096幅のモデルでAdamWの状態込み約0.6GB）。再開・評価・第2段階の前段はこのファイルだけを読みます。途中のcheckpointも残したい場合は `--set train.keep_checkpoints=true` を指定すると、`checkpoints/step-XXXXXXX.pt` のコピーも保存します（1回の保存ごとに約0.6GB増えます）。既存の `step-*.pt` は自動では削除しません。

## 検証（`pytest`）

- 正規化→逆変換で入力を復元できる／統計がtrain行の直接計算と一致する
- validationデータを変えてもtrain統計が変わらない（リーク検出）
- SIGRegが式を複素数・float64で直接計算した参照実装と一致する
- 平均シフト・分散変更・collapse・rank-1を標準Gaussianと区別できる
- 再構成・SIGRegの両方からencoderへ有限の勾配が流れる
- λ=0でSIGRegの計算と乱数消費を完全に省略する
- 条件間で初期重みとデータ順序が一致する
- checkpoint再開でデータ順序・射影乱数・学習率が継続し、中断なしの学習とパラメータが一致する
- validationがtrain側の乱数状態を変更しない
- 評価バッチがsplit全体から固定seedで抽出され、重複なく複数shardを混ぜ、呼び出しごとに同一になる
- 系列内で強く相関する（周辺は標準Gaussianの）データで、保存順バッチのSIGRegは参照値より大きく、シャッフル評価では参照値近くになる
- validation shardが1個なら系列単位で分割し、分割不能・必須splitが空なら学習前に停止する
- 学習に影響する設定を変えた再開を拒否し、ログ・評価設定の変更は許可する
- 評価バッチの各行のshard・系列・系列内位置が、両形式で実データと一致する
- 先頭トークンに入れた外れ値を検出して帰属させ、除外・上位除外後の共分散が参照値に戻る
- 全分散を保った低ランク化は、除外・上位除外後も低ランクとして残る
- 上位除外のモーメント差し引きが、残りのサンプルで計算し直した値と一致する

## 構成

| ファイル | 内容 |
|---|---|
| `src/sae_jepa/models.py` | `DenseSIGRegAE`、`build_model` |
| `src/sae_jepa/sigreg.py` | 射影生成・Epps–Pulley損失・参照実装 |
| `src/sae_jepa/config.py` | データ・モデル・SIGReg・最適化・評価の設定 |
| `src/sae_jepa/train.py` | 学習ループ・勾配診断・checkpoint/再開 |
| `src/sae_jepa/normalization.py` | train統計・PCA whitening |
| `src/sae_jepa/data.py` | manifest読み込み・split・再開可能なbatch iterator |
| `src/sae_jepa/evaluate.py` | 密な表現専用の評価 |
| `src/sae_jepa/reporting.py` | 集計レポート |
| `src/sae_jepa/frontends.py` | 第2段階用の固定前段（Raw/PCA/Dense） |
| `configs/` | Dense-AE / Dense-SIGReg-AE |
| `scripts/` | sweep・評価・スモークテスト |
