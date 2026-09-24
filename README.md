# SAE-JEPA

LLM残差ストリームに対し、**Top-K SAEの前段として「再構成＋SIGReg」で密なGaussian化表現を作る**ための実験コードです。
[JEPA-SAE](https://github.com/fumin0ri/JEPA-SAE) の活性抽出フォーマット（`sr-extract-pile`）をそのまま読み込みます。

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

JEPA-SAEの `sr-extract-pile` で作成した `manifest.json`（`shared-residual-sequence-shards-v2`）を使います。
各系列の `[burn_in_tokens, valid_length)` の位置を独立サンプルとして扱います（`data.skip_burn_in`）。

- train: manifestの `train` shard。**正規化統計はtrainからのみ計算**します。
- validation / test: manifestに `test` があればそれを、なければ `validation` shardの後半半分をtestとして予約します（`data.test_split=auto`, `holdout_test_fraction=0.5`）。validation shardが1個しかない場合は、そのshard内で**系列単位**に分割します（`path#rows=start:stop`）。分割できない場合（系列が1本など）は学習前にエラーで停止します。shard・系列範囲の重複もエラーになります。
- 学習開始時に `eval.required_splits`（既定 `[validation, test]`、sweepでは `EVAL_SPLITS` から設定）の各splitが空でなく、評価バッチを1つ以上作れることを確認します。validationだけで評価する場合は `data.test_split=none` と `eval.required_splits=[validation]`（sweepでは `EVAL_SPLITS=validation`）を指定してください。

## 実行

### Pilot sweep（Dense-AE と Dense-SIGReg-AE）

```bash
ACTIVATION_MANIFEST=runs/shared-data/pile-activations/manifest.json \
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

**評価バッチ**：同じ文書内の連続トークンは強く相関するため、保存順のバッチでは各トークンの分布がGaussianでもバッチ単位のSIGRegが大きく出ます。そこで評価専用の固定seed（`eval.sample_seed`）で**split全体（全shard・全系列）から** `eval.batches × eval.batch_size` 個の位置を非復元抽出し、ランダムな順序でバッチに割り当てます（`eval.batches=0` なら全フルバッチ）。抽出はデータとseedだけで決まるため、全λ・全checkpointが同一のバッチで評価されます。読み込みはmemory mapで行い、必要な行だけを読みます。

すべてのGaussian化指標に、**同じバッチ数・バッチサイズ・射影数の厳密なGaussianサンプルでの参照値**（`reference/...`）を併記します。
dead featureや「正なら発火」などの疎モデル用指標は密モデルには適用しません。

レポートは (FVU, held-out SIGReg) のPareto frontを示すだけで、最低SIGRegのcheckpointを自動的に勝者にはしません。

## checkpoint内容

重み、optimizer、scheduler、乱数状態（torch/cuda/SIGReg generator/データiterator）、`mu, s`、データmanifest記録（fingerprint・split別shard）、解決済みconfig、SIGReg規約。

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
