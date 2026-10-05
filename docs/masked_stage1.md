# Stage 1: full / masked activation consistency + SIGReg

既存の再構成＋SIGRegと比較する、decoderなしのStage1です。
LLMは再学習せず、保存されたトークンごとの残差activationを使います。

```text
x = (h - train_mean) / train_scalar_std
x_mask = Bernoulli(1 - p) * x
y_full = G(x)
y_mask = G(x_mask)
L = mean((y_mask - y_full)^2)
    + lambda * (SIGReg(y_full) + SIGReg(y_mask)) / 2
```

- Gは共有のLinear → GELU → Linear。次元の既定は4096 → 4096 → 4096。
- maskは各トークン・各座標で独立に、提示のたびに再生成します。
  隠すのは残差の座標で、トークン位置ではありません。pは隠す確率であり厳密な個数ではありません。
- train統計で正規化した後に0を入れるので、元空間ではtrain平均で置換する操作です。
  dropoutのような1/(1-p)の拡大、mask indicator、tokenごとの正規化はありません。
- 両方へ勾配を流し、stop-gradient・teacher・EMAは使いません。
- SIGRegは両viewで同じ射影を使い、viewごとにNトークンのECFを計算してから平均します。
  fullとmaskedを2Nサンプルとして混ぜません。射影はstepごとに再生成します。
- SIGRegの数値積分は既存AEと同じ[-5,5]・17点、N係数あり、float32です。
- 入力統計・split・先頭位置除外・データ順序・初期encoderは既存AEと共通にできます。
  AEからencoderを初期化するのではなく、同じseedの初期値を使います。
- mask RNGはCPU上の専用generatorで、モデル・データ・SIGReg乱数とは独立です。
  checkpointにmask RNGを含め、中断再開を再現します。mask率などの変更再開は拒否します。

## 実行

実験環境（`/home/iwasaki`）のactivation manifestは次のパスです。
他の環境では保存先に合わせて変更してください。

```bash
export ACTIVATION_MANIFEST=/home/iwasaki/LeJEPA-SAE/data/the-pile/pythia-6.9b/layer-16-ctx1024-100m/manifest.json
```

このデータでmask率0.1／0.5のStrong SIGReg＋covariance実験を順番に実行する例
（リポジトリ直下で実行）:

```bash
RUN_ROOT=runs/masked-cov-beta1 \
MASK_PROBS="0.1 0.5" WEIGHTS="0.1" \
STEPS=100000 SEED=42 PARALLEL=1 \
bash scripts/stage1_masked_sweep.sh \
  --set covariance.weight=1.0 \
  --set covariance.sketch_dim=64 \
  --set model.init_output_variance=1.0 \
  --set optim.lr=0.00003
```

上の`export`を同じシェルで先に実行してください。batch512、ZCAなし、初期出力の
中心化・分散調整ありです。beta=1は探索の出発点です。covなしの対照は
`RUN_ROOT`を別名にして、`covariance.weight=0`に変更します。

汎用的な実行例:

```bash
pip install -e '.[dev]'
ACTIVATION_MANIFEST=/path/to/manifest.json \
RUN_ROOT=runs/stage1-masked-100k \
MASK_PROBS="0.25 0.5" WEIGHTS="0.001 0.003" \
STEPS=100000 SEED=42 \
bash scripts/stage1_masked_sweep.sh
```

既定で先頭位置0を除外し、100k updates、batch512、末尾50%のlinear decayです。
`NORMALIZATION=/path/to/normalization.pt`で既存のtrain統計を再利用できます。
データ・除外条件が一致しなければ停止します。評価は既定でvalidationのみです。
λは探索の出発点であり、再構成損失から移した最適値ではありません。

短い動作確認（batch数を減らした結果を本実験の結論には使わないでください）:

```bash
ACTIVATION_MANIFEST=/path/to/manifest.json RUN_ROOT=runs/masked-smoke \
MASK_PROBS=0.25 WEIGHTS=0.001 STEPS=100 \
bash scripts/stage1_masked_sweep.sh \
  --set optim.warmup_steps=10 --set eval.batches=4 --set train.validation_every=0
```

単独学習・再開・評価:

```bash
sj-masked train --config configs/masked_sigreg.yaml \
  --set data.activation_manifest=/path/to/manifest.json \
  --set train.output_dir=runs/masked-single --set masking.probability=0.25
# 同じ引数でlatest.ptから自動再開。総stepsなどの学習条件は変更不可。
sj-masked evaluate --checkpoint runs/masked-single/checkpoints/latest.pt \
  --split validation --device cuda
# 条件決定後の最終test（test splitが存在する必要があります）
sj-masked evaluate --checkpoint runs/masked-single/checkpoints/latest.pt \
  --split test --device cuda
sj-masked report --run-root runs/masked-single
```

`python -m sae_jepa.masked ...`も同じです。
評価の`--set`はeval.*のみ許可します。mask率はcheckpointの設定を使います。

## Strong SIGReg + sketched covariance

`covariance.weight=beta > 0` で、既存の目的に次を追加します。

```text
R = reduced_QR(randn(d_latent, k)).Q   # R.T @ R = I_k; every step
z_view = y_view.float() @ R
z_view = z_view - z_view.mean(0)
C_view = z_view.T @ z_view / (B - 1)
cov_view = mean((C_view - I_k)^2)
L += beta * (cov_full + cov_mask) / 2
```

両枝は同じRを使い、中心化・共分散計算は枝ごとに行います。
`masking.enabled=false` ではfull枝だけに追加します。decoderは追加しません。
計算はautocastを無効にしたfloat32です。Frobeniusノルムの二乗を **k²で割る**
規約であり、SIGRegのようなB係数は掛けません。平均の制約はStrong SIGRegが担います。

```bash
sj-masked train --config configs/masked_sigreg.yaml \
  --set data.activation_manifest=/path/to/manifest.json \
  --set train.output_dir=runs/masked-cov-beta1 \
  --set masking.probability=0.1 --set sigreg.weight=0.1 \
  --set covariance.weight=1.0 --set covariance.sketch_dim=64 \
  --set model.init_output_variance=1.0
```

beta=1は動作例であり、最適値ではありません。初期出力調整も任意です。
入力は既定のscalar正規化で、ZCAを自動適用しません。
対照は同条件で `covariance.weight=0` とし、別の出力ディレクトリで学習します。
既定のweight=0では共分散損失も射影生成も行わず、従来の学習挙動を維持します。
`1 <= k <= d_latent`、`k < optim.batch_size`、`k < eval.batch_size` が必要です。

射影用の専用CPU RNGをcheckpointへ保存します。データ順序・mask・SIGRegのRNGは
消費しません。学習設定を変えた再開は拒否します。共分散設定を持たない旧checkpointは、
既定の無効状態で再開できます。

学習ログは `covariance`、`covariance_full`、`covariance_masked`、
`covariance_weighted` と `grad_rms_y/{full,masked}/covariance_weighted` を記録します。
mask無効時はmasked項を出しません。betaは重み付きSIGReg・consistencyとの勾配RMSを
比較して調整してください。

評価では専用seedの固定直交射影を用い、batchごとの損失平均を
`{gaussian,masked,reference}/sketched_covariance` に記録します。
これは評価全体を集計した既存の共分散・固有値指標とは別です。
重み・k・評価seedも保存し、reportにはbetaとkを表示します。

有限batchでは、独立な標準ガウスでも期待値は `(k+1)/(k*(B-1))` です。
これを `covariance_gaussian_expected_value` として併記します。
ノイズフロアの減算やバイアス補正は行いません。ガウス母共分散 `a I_k` に対して
この損失単独が好む分散は `a=(B-1)/(B+k)` なので、平均分散の縮小にも注意します。
（k=256, B=512 では a≈0.665 で、βが支配的なrunの平均分散はこの値に近づきます。）

### split-half推定量（`covariance.estimator=split_half`）

上の縮小を避けるには `--set covariance.estimator=split_half` を指定します。

```text
C_1 = Cov(z[:B//2]);  C_2 = Cov(z[B//2:])   # 半分ずつ中心化、分母 n-1
cov_view = mean((C_1 - I_k) * (C_2 - I_k))
```

2つの半分が独立なら期待値は母集団の `||Cov - I_k||_F^2 / k^2` に一致し、
最小点は Cov = I（縮小なし）、N(0,I) での期待値は0です（負の値も取り得ます）。
勾配も母集団損失の不偏推定です。学習batchはshard window内でシャッフル済みなので
前半／後半を使います。分散はplug-inより大きくなります。
既定は従来の `plugin` で、旧checkpointはそのまま再開できます。推定量を変えた再開は拒否します。
評価JSONには `covariance_estimator` を記録し、`sketched_covariance` も同じ推定量で計算します。

### 相関のみのsplit-half（`covariance.estimator=split_half_corr`）

`split_half` でも、相関 R が残る限り `||s² R - I||²` は `s² = k / ||R||_F² < 1` で最小になり、
βを上げると分散が縮みます（λ=0.01, 10k で β=100/300/1000 の平均分散 0.87/0.82/0.75）。
`split_half_corr` は各半分の相関行列の非対角成分だけを使います。

```text
r_h = Corr(z_half)                          # 半分ずつ中心化
cov_view = sum_{i!=j} r_1[i,j] * r_2[i,j] / k^2
```

方向ごとのスケールには依存しないので分散を動かさず、分散を1に合わせるのはSIGRegだけになります。
座標が独立な分布での期待値は0です（負の値も取り得ます）。非零の相関は半分のサイズnに対しO(1/n)だけ小さく推定されます。
k²で割る規約は他の推定量と同じです。対角項がないぶん、同じβでも値は小さくなります。

### βのwarmup（`covariance.warmup_steps`）

`covariance.warmup_steps=N > 0` で、βを step 0 の0から step N の満値まで線形に上げます
（`beta_t = beta * min(1, t/N)`）。既定の0は従来どおり最初から満値です。
相関の勾配は 1/σ に比例するため、出力分散が小さい学習初期に強いβを掛けると
Adamの更新が相関項に支配され、SIGRegが分散を育てられません
（split_half_corr, λ=0.01, 10k で β=300/1000 の分散は 0.22/0.05 に留まりました）。
warmup中も射影は毎step生成するのでRNG系列は変わりません。学習ログには
`covariance_weight_effective` を記録します。warmup_stepsを変えた再開は拒否します。
検証にはheld-out SIGReg・全共分散のPR／有効ランク／最大固有値・ノルム尾部も使ってください。

## 指標と読み方

SIGReg単体での入力ZCA whitening対照は、[入力whitening実験](input_whitening.md)を参照してください。

### 同じサンプルの入力・出力ノルムを調べる

既存checkpointを再学習せず、評価時に `--norm-diagnostics` を追加します。
maskedとSIGReg-onlyの両方に対応しています。

```bash
run_dir=runs/stage1-sigreg-only-100k/seed-42
sj-masked evaluate --checkpoint "$run_dir/checkpoints/latest.pt" \
  --split validation --device cuda --norm-diagnostics \
  --norm-outlier-fraction 0.01 \
  --output "$run_dir/eval-validation-norms.json"
```

通常の評価JSON・spectraに加えて、次を保存します。

- `eval-validation-norms-norm-samples.jsonl`: 全評価サンプルの対応表。
  `h_sqnorm_per_dim` は生activation、`x_sqnorm_per_dim` はtrain平均・共通スカラー
  で正規化した入力（入力whiteningを有効にした場合は、その変換後）、`y_sqnorm_per_dim` はfull出力の二乗ノルム/各ベクトルの次元。
  masked学習の場合は `masked_x_sqnorm_per_dim` と `masked_y_sqnorm_per_dim` も保存。
  `sample_index` は評価順、`entry` はsplitのshard一覧へのindex、`shard` はmanifest内のentry、
  `sequence` と `position` はshard内のsequence indexとその中のtoken位置（0始まり）。
  token文字列やIDの復元は行いません。
- `eval-validation-norms-norm-summary.json`: Pearson/Spearman相関、出力上位1%と
  入力上位1%の重なり、出力上位群/残りの入力・出力ノルム分布、上位32件の対応表、
  エネルギー比、上位群が全出力二乗ノルム総和に占める割合、評価条件を保存。

相関は二乗ノルム/次元に対するものです。Spearmanは同順位を平均順位として扱い、
一定値の列の相関はnullです。上位件数はceil(N×fraction)（最低1件、最大N−1件）で、
同じノルムではsample_indexの小さい順に選びます。
`y_over_x_energy_ratio` は `(||y||²/d_latent)/(||x||²/d_in)` です。
入力0の場合はnullとし、比の集計から除外して件数を記録します。
非線形モデルの作用素ノルムではありません。出力上位サンプルの入力も巨大か、
通常の入力が出力で増幅されたかを確認するための指標です。

`--norm-outlier-fraction` は対応診断の群分けにだけ使います。
共分散のtrim率は従来どおり `--set eval.outlier_fraction=0.01` で別途指定します。
診断は同じ評価forwardからスカラーだけを収集し、追加のmask生成やサンプリングをしません。
元の評価値・乱数列は変えず、activation全体を保持しないため追加メモリはサンプル数に比例します。
現在のCLIのsplitはvalidation/testです。train分布の診断はこの機能には含みません。

巨大ノルムの元トークン、保存値とloaderの一致、元LLMでの再計算は
[activation検証手順](activation_verification.md)の`sj-verify-activations`を使用します。

### SIGReg単体の切り分け

`configs/sigreg_only.yaml`は`masking.enabled=false`とし、full入力のSIGRegだけで
同じencoderを学習します。再構成、一致損失、masked branchは計算しません。
λ=1で`L=SIGReg(G(x))`です。ほかの損失との重み比を探索する実験ではありません。
既存のmask実験と同じ入力正規化を使い、別runとして開始してください。

```bash
sj-masked train --config configs/sigreg_only.yaml \
  --set data.activation_manifest=/path/to/manifest.json \
  --set data.normalization_path=runs/stage1-masked-100k/normalization.pt
sj-masked report --run-root runs/stage1-sigreg-only-100k
```

学習終了時にvalidationのGaussian性・共分散・trimmed診断を自動保存します。
`sj-masked evaluate`でも再評価できます。SIGReg単体ではmasked側と一致損失の指標は
存在せず、レポートの該当欄は空欄です。旧checkpointは既定のmask有効として読み、
mask有効/無効を切り替えた再開は拒否します。

### 共通の評価指標

- `consistency/mse`: fullとmaskedの一致。小ささだけでは情報保持を保証しません。
- `gaussian/*`: 下流に使う**full入力**の表現のGaussian性・共分散。
- `masked/*`: masked入力の同じ指標。
- `reference/*`: 同じ標本数・batch sizeの標準Gaussian参照。
- 詳細評価ではheld-out/diagnostic SIGReg、W2、分位点、共分散固有値、
  有効ランク、participation ratio、ノルム分位点を保存します。
- 各viewと参照について高ノルム標本を除いた共分散も保存します。
  除去件数は`trimmed/count`に記録され、eval.outlier_bufferで上限を設けます。
- `*-spectra.pt`にfull/masked/referenceとtrimmedの固有値を保存します。
  `report/validation.md`に比較表、`validation_spectra.png`に両viewのスペクトルを出します。
- 学習ログには両viewの一致損失・重み付きSIGRegの勾配RMSを記録します。
- 評価maskと射影は固定seed。評価は学習RNGを消費しません。

**full側も更新されるので、元activationの全情報を保持する保証はありません。**
一定表現へのcollapseを防ぐため、このコマンドはλ>0を要求します。
SIGRegが低くても低ランク化し得るため、スペクトルを必ず参照してください。

## 後段との境界

この実験ではStage1 decoderを学習・保存しません。再構成FVUを出したり、
未学習decoderを使ったりはしません。既存`sj-stage2`、`sj-probe`、
`sj-evaluate-dense`、`sj-diagnose-dense`の再構成付き経路には直接渡せません。
checkpointを渡すと明示的に拒否します。集計は`sj-masked report`を使ってください。

後段SAEへの入力は固定encoderの`encode_dense(h)`（full入力）です。
元activationでの再構成評価まで拡張する場合は、encoderを固定してtrain splitだけで
別のreadout decoderを学習し、そのdecoderも固定してSAEを比較する追加工程が必要です。
今回の実装範囲はStage1の仮説検証で、この追加学習工程は含みません。
