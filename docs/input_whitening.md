# SIGReg-only：入力whiteningの対照実験

スカラー正規化だけの入力と、固定ZCA whiteningを加えた入力を比較します。
モデル・初期重み・学習サンプル順・SIGRegの投影乱数・学習予算は同じにします。
初期重みを共有しても、入力が変わるため学習後の重みは当然異なります。

## 変換

```
x = (h - train_mean) / train_scalar_scale
z = (x - whitening_center) @ U @ diag((eigenvalues + epsilon)^(-1/2)) @ U.T
y = encoder(z)
```

trainからのみ共分散を推定し、変換は学習中に更新しません。ZCAなので次元は削減せず、
元の座標系に戻します。これはモデル出力を後からwhiteningする実験ではありません。
共分散はfloat64で中心化・集約し、変換行列をfloat32で保持します。
適用時の行列積ではbf16 autocastとTF32を無効にします。エンコーダ本体の精度設定は変えません。

正則化により、fit集合での変換後の固有値は `eigenvalue / (eigenvalue + epsilon)` です。
小さい固有値を過剰に増幅しない代わりに、共分散が厳密なIになるとは限りません。
validationでもIになる保証はないため、入力自体のスペクトルを確認してください。

今回の実装は **SIGReg-only専用** です。`masking.enabled=true`や再構成AEとの組合せは
エラーにします。whitening後の座標マスクは元の実験と意味が変わるためです。
元データの先頭除外設定は維持しますが、新たなノルム外れ値の除外は追加しません。

## 実行

リポジトリのルートで実行します。

```bash
git pull --ff-only
pip install -e .

manifest=/home/iwasaki/LeJEPA-SAE/data/the-pile/pythia-6.9b/layer-16-ctx1024-100m/manifest.json
normalization=runs/stage1-masked-100k/normalization.pt
whitening=runs/input-whitening/train-zca-262144-eps1e-4.pt

sj-fit-input-whitening \
  --activation-manifest "$manifest" \
  --normalization "$normalization" \
  --skip-leading-positions 1 \
  --maximum-positions 262144 \
  --sample-seed 1729 \
  --epsilon 1e-4 \
  --chunk-size 2048 \
  --device cuda \
  --output "$whitening"

ACTIVATION_MANIFEST="$manifest" \
NORMALIZATION="$normalization" \
INPUT_WHITENING="$whitening" \
RUN_ROOT=runs/stage1-input-whitening-100k \
STEPS=100000 SEED=42 \
bash scripts/stage1_input_whitening_compare.sh
```

この例はtrain全体の利用可能な位置から262,144件を一様・非復元抽出してfitします。
trainがそれより少ない場合は全件を使います。先頭の数shardだけに偏ったprefix抽出ではありません。
`--maximum-positions 0`は全train位置を使用します。全件fitは大きな共分散計算を伴うため、
まず上限付きfitの入力診断を確認してください。`--device cpu`も使用できます。

比較スクリプトは、scalarとzcaの2条件を順番に学習します。既定では潜在次元4096、
batch512、投影256、λ=1、学習率1e-4、100k step、maskなしです。
元のSIGReg-only条件とそろえています。保存先は `scalar/seed-42` と `zca/seed-42`。
`--set data.input_whitening_path=...`を指定した単独学習も可能です。

## 保存物・評価

whitening fitは以下を保存します。

- `.pt`: 平均・正則化・ZCA行列・固有値・train由来を確認するメタデータ。
- 同名`.json`: fit件数、使用shardごとの件数、抽出seed、正則化、変換前と期待される変換後の固有値。

学習checkpointには変換を埋め込みます。offline評価時は元のwhitening `.pt` がなくても
checkpointだけから変換を復元します。学習再開時は、指定したfitファイルの変換が
checkpointと同じであることを確認し、異なれば停止します。再開用にはfitファイルを保持してください。
旧checkpointでは新オプションは既定で無効となり、元の挙動を維持します。

比較スクリプトでは `eval.input_diagnostics=true` を有効にします。

- `input_transform`: scalar または zca。
- `encoder_input/*`: **エンコーダに実際に入れた表現**の平均・共分散・有効ランク・PR。
- `gaussian/*`: 学習済みエンコーダの出力。従来と同じ指標。
- `*-spectra.pt`内の`encoder_input`: 入力共分散の固有値。

まず「zca入力の共分散がscalar入力よりIに近いか」、次に「出力の異方性が改善するか」を
分けて確認します。入力の等方性が改善していない場合、実験が入力条件を十分に変えられて
いないため、出力だけから仮説を否定できません。

`--norm-diagnostics`も併用可能です。この場合、`h`は生activationのままですが、`x`は
**whiteningを含めた実際のエンコーダ入力**です。`provenance.input_transform`で区別します。
scalar条件のxとzca条件のxは異なる座標変換後の量であることに注意してください。

whiteningによって出力の等方性が改善しても、情報保存・SAE/probe性能の改善は別の検証です。

## 学習中の出力ランク記録と、出力分散の初期化

### 学習中のPR・有効ランク

masked / SIGReg-onlyエンコーダの学習中validation（`train.validation_every`ごと）でも、
出力の共分散を集計します。`metrics.jsonl`の`validation`に次が入ります。

- `gaussian/cov_effective_rank`, `gaussian/cov_participation_ratio`,
  `gaussian/cov_eig_max`, `gaussian/cov_eig_min`, `gaussian/cov_fro_dev` など
- mask有効時は`masked/cov_*`も同様
- Gaussian参照は固定なので学習中には共分散を計算しません

集計は`train.validation_batches`分（既定16 batch = 8,192件）です。最終評価（64 batch）より
件数が少ないので、絶対値ではなく推移の比較に使ってください。新規学習では
**step 0（初期化直後、更新前）にもvalidationを1回記録**します。
`eval.training_covariance=false`で無効化できます（再開時に変更可能）。

### `model.init_output_variance`

既定の初期化では、SIGReg-onlyの初期出力分散が約0.035と小さく、学習初期は
スケールを上げる勾配が支配的になります。`model.init_output_variance=1.0`を指定すると、
エンコーダ構築後（ZCA使用時はwhitening適用後）に、最後のLinear層を次のように変更します。

```text
y0 = G(x)                       既定初期化の出力
y  = gain * (y0 - mean(y0))     gain = sqrt(target / mean_per_dim_var(y0))
```

- 推定には、train splitから専用seedで抽出した固定サンプル（8 batch × `eval.batch_size`）を使います。
  学習のデータ順序・モデル・SIGReg・maskの乱数は変わりません。
- 第1層は変えません。初期出力の固有値スペクトルの**形**は既定初期化と同じで、
  全体スケールと平均だけが変わります。
- `gain`・推定前の分散・件数は`sigreg_convention.json`の`output_init`に保存されます。
- 既定値0は従来の初期化です。旧checkpointの再開挙動は変わりません。

比較スクリプトは、ZCA入力で既定初期化と出力分散1初期化の2条件を学習します。

```bash
ACTIVATION_MANIFEST=/path/to/manifest.json \
NORMALIZATION=runs/stage1-masked-100k/normalization.pt \
INPUT_WHITENING=runs/input-whitening/train-zca-262144-eps1e-4.pt \
bash scripts/stage1_output_init_compare.sh
```

`DECAY_FRACTION`（既定0.5）、`INIT_VARIANCE`（既定1.0）、`STEPS`、`SEED`、`RUN_ROOT`で
条件を変更できます。step 0の有効ランクがどちらも同程度（約1,900）で、既定初期化だけが
学習初期に大きく下がるなら、ランク低下は初期のスケール上げ段階で生じていると判断できます。
