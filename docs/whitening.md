# PCA whitening対照実験

SIGReg前段のsparse probing結果を解釈するための対照です。
[Data Whitening Improves Sparse Autoencoder Learning (arXiv 2511.13981)](https://arxiv.org/abs/2511.13981)
のPCA whiteningと、これまで欠けていた**生活性のベースライン**を、同じstage-2予算で比較します。

## 問い

1. 我々の設定（Pythia-6.9B L16、16k / K=64、5.12Mトークン）でも、PCA whiteningはRawよりprobe性能を上げるか。
2. SIGReg前段の低下は、Gaussian化そのものによるのか、**SAE損失を潜在空間で測ったこと**によるのか。
3. 非線形のDense前段そのもの（λ=0）がRawより悪いか。

## 条件

| 前段 | `latent` 損失 | `original` 損失 |
|---|---|---|
| Raw `x=(h−μ)/s` | ＝originalのスケール違い（確認用） | **基準** |
| PCA whitening | whitened空間のMSE | **論文の再現**（dewhiten後のMSE） |
| Dense λ=0 | 従来のstage 2 | 固定decoderを通したMSE |
| Dense λ=0.001 | 従来のstage 2 | 固定decoderを通したMSE |

`original` 損失は `mean((D(s_y û + μ_y) − x)²)` です。`D` は固定した線形decoderで、勾配はDを通してSAEへ流れます。
Raw/PCAではDが厳密な逆写像なので、end-to-end FVUを直接最小化します。
Denseでは目標がxなので、前段自身の再構成誤差（約0.1%）の一部をSAEが補正する場合があります。
Rawの2条件は損失が定数倍（calibration尺度²）違うだけです。ただしAdamのepsとgradient clipの効き方が変わるので、完全には一致しません。

全runでstage-2設定・SAE seed・学習データ順・初期SAEが同一です。
`sj-stage2` が各runの初期ハッシュを記録し、reportとprobeは損失空間以外の設定の違いを拒否します。

## PCAの推定（`sj-fit-pca`）

- `--like` で指定したstage-1 checkpointから、データ方針（先頭除外、split）と `μ, s` をそのままコピーします。
  Raw（`sj-make-raw-frontend --like`）も同じです。
- 既定ではtrain split全体から、2,097,152トークンを固定seed（`--sample-seed 92001`）で非復元一様抽出します。
  float64で2次モーメントを積算します。`--maximum-positions 0` にするとtrain全体を使います。
  論文は約3万トークンで推定していますが、4096次元では小さい固有値が大きく偏るので採用しません。
- `ε` は既定で `1e-3 × 平均固有値` です（`--relative-epsilon`）。xは1次元あたり平均分散が1なので、ε≈1e-3になります。
  旧既定の絶対値 `1e-6` では、弱い方向が最大約1000倍に増幅されます。
  **εはprobe結果を見る前にここで固定し、結果に合わせて調整しません。**
- 出力の `pca.json` に次の診断を保存します。
  - 固有値スペクトルの要約：条件数、最大whitening倍率、participation ratio、上位k成分の累積分散、ε未満の固有値数
  - 交互バッチで分けた2半分から推定した上位k部分空間の一致度（`subspace_overlap`、1＝一致）

## 実行

```bash
STAGE1_ROOT=runs/stage1-skip-lead-100k-v2 \
RUN_ROOT=runs/stage2-whitening \
bash scripts/whitening_sweep.sh
```

- 既定の比較対象はRaw、PCA、Dense λ=0、Dense λ=0.001です。これらを `latent original` の両損失で学習し、計8 runになります。
  追加の設定：`WEIGHTS`、`LOSS_SPACES`、`PCA_POSITIONS`、`PCA_RELATIVE_EPSILON`、`FRONTEND_DIR`、`ACTIVATION_MANIFEST`、`RESUME=1`。
- `model-XX` の番号は、前段ごとに損失空間を並べた順（raw/latent, raw/original, pca/latent, …）です。
- 既存のstage-2 pilotと同じ前段・設定のrun（Dense/latent）は、旧結果の再現確認にもなります。

probeは既存スクリプトで、`STAGE2_ROOT` 以下の全 `model-*` を比較します。

```bash
PROBE_TASKS=data/probe/saebench.jsonl \
ACTIVATION_CACHE=data/probe/saebench-activations \
STAGE2_ROOT=runs/stage2-whitening RUN_ROOT=runs/probe-whitening \
bash scripts/probe_sweep.sh
```

## 判定（事前に固定）

- **主指標**：AG News以外も含む各タスクの**validation**での、データセット平均Top-1。
  補助指標：Top-2/5、`--standardize` での再集計、元空間FVU、L0、非発火率、発火集中度。
- **次の段階へ進む基準**：P-original − Raw-original ≥ +2pp で、かつ同じ向きのデータセットが8つ中6以上なら、SAE seed 43/44を追加する。
- **testの扱い**：testはseedを追加した最終比較で1回だけ評価する。AG Newsのtestは既に見ているため、最終判断には使わない。

## 結果の読み方

| 結果 | 解釈 |
|---|---|
| PCA-original > Raw-original | 論文の効果が本設定でも出ている。SIGRegとの差は、分布の形か非線形前段の違いに絞れる |
| PCA-original ≤ Raw-original | 学習予算（論文の約1%）、モデル規模、K、学習率の違いを疑う。延長runで確認する |
| Dense-original ≫ Dense-latent | SIGReg前段の敗因の一部は損失空間だった |
| Raw > Dense λ=0 | 非線形前段そのものがprobe性能を下げている |
