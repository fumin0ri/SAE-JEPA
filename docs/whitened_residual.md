# PCA whitening上の恒等残差AE（`whitened_residual_ae`）

SIGRegによる非線形のガウス化が、線形whitening（PCA baseline）に何かを上乗せするかを調べる前段です。
従来の`dense_sigreg_ae`は白色化そのものをencoderで一から学ぶため、Stage1でFVUを約10%失っていました。
そのため、PCAとの差に「高次のガウス化の効果」と「情報損失」が混ざっていました。
この前段はPCA baselineと完全に同じ表現から学習を始めます。そのうえで、再構成損失によって情報を保ったまま、SIGRegで高次の分布形状だけを動かします。

## モデル

```text
x     = (h - mu) / s                  baselineと同じtrain-only正規化
z     = (x - c) W                     baselineのPCA（固定、float32・TF32なし）
y     = z + G_theta(z)                G: Linear -> GELU -> Linear、最後のLinearをゼロ初期化
x_hat = D_phi(y)                      Linear、初期値はwhiteningの厳密な逆変換
h_hat = s * x_hat + mu

L = mean((x_hat - x)^2) + lambda * SIGReg(y)   （covariance項も従来どおり指定可能）
```

- step 0では`y = z`と`x_hat = x`が数値誤差の範囲で成り立ちます。step 0のvalidationがPCA baselineの値になります。
- `D_phi`は学習します。Stage2の前段FVUは、このdecoderでの再構成誤差です。
- 初期化で乱数を追加消費しません。同じseedの`dense_sigreg_ae`と、第1層の初期重みとデータ順序が一致します。
- `model.d_latent == d_in`が必要です。

## 白色化の指定（`data.input_whitening_path`）

次のどちらかを指定します。

- `sj-stage2 prepare-baselines`が作った前段（`runs/baseline-raw-pca-64k-100k/frontends/pca.pt`）
  - PCA baseline SAEと**同じ行列**を使います。比較にはこちらを推奨します。
  - ファイルのchecksumも確認します。
- `input_whitening.fit`が作ったZCA/PCAのfitファイル

`data.normalization_path`には、白色化を作ったときと同じ正規化を指定します。
正規化の値、manifest、train shard、先頭除外が一致しないと、学習前にエラーで停止します。
既存のPCA baselineは`runs/stage1/rec-sigreg-cov/normalization.pt`（`skip_leading_positions=0`）で作られています。

checkpointには変換を埋め込みます。評価・Stage2・probeでは元のファイルは不要です。
再開時は、指定したファイルの変換がcheckpointと一致するか確認します。

## 実行

```bash
pip install -e .
ACTIVATION_MANIFEST=/home/iwasaki/LeJEPA-SAE/data/the-pile/pythia-6.9b/layer-16-ctx1024-100m/manifest.json \
NORMALIZATION=runs/stage1/rec-sigreg-cov/normalization.pt \
WHITENING=runs/baseline-raw-pca-64k-100k/frontends/pca.pt \
WEIGHTS="0.001 0.01" SAE_SEEDS="42" \
PROBE_TASKS=data/probe/saebench-8.jsonl PROBE_ACTIVATIONS=runs/downstream-seeds/probe-cache \
bash scripts/stage1_whitened_residual.sh
```

1. 各λでStage1を学習します。既定は100k step、末尾50% decay、lr 1e-4、seed 42です。
   - 出力は`RUN_ROOT/stage1/lambda-*/seed-42`、レポートは`RUN_ROOT/stage1/report/`です。
2. 全λのcheckpointに、Raw/PCA baselineと同じSAEを付けて学習します。
   - 辞書65,536、K=64、100k step、`reconstruction_space=activation`です。
   - 出力は`RUN_ROOT/stage2/seed-*/model-NN`で、`WEIGHTS`の順に並びます。
3. `PROBE_TASKS`と`PROBE_ACTIVATIONS`を指定すると、baselineと同じ設定でsparse probeを評価します。

`STAGE2=0`ではStage1だけを実行します。中断後は同じ環境変数に`RESUME=1`を付けて再実行します。
Stage1は`--resume auto`で自動的に再開されます。

## 判断の手順

- まずStage1の`reconstruction/fvu`を確認します。目安は≲1%です。
  - これを大きく超えるλは、PCAとの差に情報損失が混ざるため比較から外します。
  - 同時に、held-out SIGRegと有効ランクがPCA（step 0の値）からどれだけ動いたかを見ます。
- Stage2/probeでは、同じSAE seedのPCA baseline（`runs/baseline-raw-pca-64k-100k/seed-*/model-01`）と比べます。
  - 初期SAE hashと評価標本hashが一致していることを確認します。
  - 判断はmacro平均（Top-1/2/5）とend-to-end FVUで行います。データセット別のTop-1差は学習の細部で入れ替わるため、根拠にしません。
- λ=0（`WEIGHTS="0 ..."`）は、PCAに学習可能なdecoderと残差branchを加えただけの対照です。PCA baselineとの差が、λ>0の差より十分小さいことを確認できます。
