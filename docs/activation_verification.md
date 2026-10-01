# 巨大ノルムactivationのトークン・保存・再計算検証

`sj-verify-activations` は、`sj-masked evaluate --norm-diagnostics` の結果を起点に、
出力が巨大になったサンプルを元のLLM入力へ戻して検証します。
Stage1の再学習は行わず、元のactivationファイルも変更しません。
現在の対象はLeJEPA-SAEのflat safetensors形式です。

## 必要なもの

- `*-norm-summary.json` と `*-norm-samples.jsonl`。
- 診断に使ったStage1 checkpoint（summary内のパスを既定で使用）。
- 元のactivation manifestとshard。`token_ids`がなければトークン復元・再計算は不可。
- decodeにはtokenizer、再計算には元LLMの重み。

```bash
git pull --ff-only
pip install -e '.[verify]'
```

## 1. GPUを使わず、保存データとトークンを確認

```bash
run_dir=runs/stage1-sigreg-only-100k/seed-42

sj-verify-activations \
  --summary "$run_dir/eval-validation-norms-norm-summary.json" \
  --tokenizer EleutherAI/pythia-6.9b \
  --min-output-energy 10 \
  --max-outliers 64 \
  --controls 8 \
  --matched-per-token 2 \
  --output "$run_dir/activation-inspection"
```

既定ではHugging Faceのローカルcacheだけを使用します。必要なファイルがなければ、
`--allow-download`を追加するか、`--tokenizer`にローカルtokenizerディレクトリを指定します。
token IDだけを確認する場合は`--tokenizer`を省略できます。

選択条件は出力の **二乗ノルム/d > 10** です。今回のSIGReg-only結果では52件が該当しますが、
他モデルで同じ件数になるとは限りません。上位64件までを検証します。
残りのサンプルから固定seedで通常例8件を取り、選択された巨大群の各token IDについて
通常例も最大2件ずつ追加します。同一tokenの通常例がなければ、0件として報告します。
token頻度の母集団は元データ全体ではなく、norm-samplesの評価サンプル集合です。

公式`safetensors` readerで対象activationを読み、学習用`DataSource.gather`が返す値と比較します。
manifestのoffset、sequence index、token position、先頭位置除外を用いた逆対応も検証します。
独立したreaderを使いますが、両者は同じmanifestの位置情報に依存するため、manifest自体の
系統的な誤りまでこの比較だけで否定できるわけではありません。

保存値の二乗ノルムが既存ログと一致するかも調べます。checkpoint、分割、データfingerprintを
確認し、不一致があれば別データを混ぜた検証として続行しません。

## 2. 元モデルでactivationを再計算

出力先は検査ごとに新しいディレクトリにします。次は**生成時にpaddingなしのsequenceを
全位置有効として処理していたことを確認できた場合**の例です。

```bash
run_dir=runs/stage1-sigreg-only-100k/seed-42

sj-verify-activations \
  --summary "$run_dir/eval-validation-norms-norm-summary.json" \
  --tokenizer EleutherAI/pythia-6.9b \
  --recompute \
  --dtype bfloat16 \
  --device cuda \
  --attention-mask all-ones \
  --attn-implementation eager \
  --min-output-energy 10 \
  --max-outliers 64 \
  --controls 8 \
  --matched-per-token 2 \
  --output "$run_dir/activation-recompute"
```

初回は`--max-outliers 4 --controls 4`で小さく確認することもできます。

`--attention-mask`の既定は`stored`です。shard内の`attention_mask`を使い、存在しなければ
再計算を停止して不足情報を記録します。`all-ones`は明示的な仮定であり、自動推測しません。
`position_ids`が保存されていればそれを使い、なければモデルの既定を使います。
特殊なposition設定や文書間のblock mask、cross-segment KV cacheを使った生成は、この再計算では再現できません。

モデル名・revision・layer・hookはmanifestを使用します。今回は`block_output:16`で、
0始まりの16番目のblockの出力です。layerのhidden-state indexや最終layer norm後の表現と
混同しないようforward hookを使います。入力は保存済みtoken IDsの**sequence全体**であり、
表示用の短いcontextをforwardしたり、decodeした文字列を再tokenizeしたりはしません。
同じsequenceの対象位置は1回のforwardでまとめて検証します。

`--model`でローカルのモデルコピー、`--revision`で当時のrevisionを指定できます。
`--dtype`と`--attn-implementation`は実際の生成条件に合わせてください。
CLI既定のbfloat16/eagerが生成時の条件だったと保証するものではありません。
元manifestがrevision=`main`しか保存していない場合、当時と同じ重みであることは確定できません。
requested/resolved revision、精度、backend、mask設定、ライブラリversionを結果に記録します。

## 出力と判定

`verification.json`に検査結果を保存します。

- `token_counts`: 巨大群のtoken IDごとの件数、評価集合内の通常例件数。
- `samples`: 元のshard/sequence/position、文脈token IDs、decode結果（指定時）、special tokenか。
- `direct_vs_loader`: 保存値と学習用loaderの比較。
- `logged_h_norm_matches`: 保存値がnorm診断時の入力ノルムと一致するか。
- `recomputed_vs_stored`: 再計算したblock出力と過去の保存値の相対L2誤差・cosine・二乗ノルム。
- `storage_cast_vs_stored`: 再計算値を過去の保存dtypeにcastした後の比較。
- `new_roundtrip`: 再計算・cast済みの値を新しいsafetensorsに保存して読み戻した比較。
- `group_errors`: 巨大群・通常群・同一token通常群それぞれの誤差集計。

`roundtrip-<sample_index>.safetensors`は診断用の新規ファイルです。実際の元データを
上書きしません。castの影響と新しい保存/読込の影響を分けて比較します。
この往復が一致しても、過去の抽出コードにバグがなかったことまでは証明できません。

固定閾値で「正常/破損」を自動判定しません。まず通常群の再計算誤差を基準にしてください。
通常群も巨大群も近く、巨大activation自体も再現されるなら、元モデルの挙動である証拠が強まります。
通常群から大きく違うなら、モデルrevision、hook、入力対応、精度、mask設定などを先に見直します。
ノルムだけ近くベクトルが違う場合も検出できるよう、相対誤差とcosineを併記します。

再計算中に情報不足・OOM等で停止した場合も、先に完了した検査結果を保存し、
`status=recomputation_incomplete`とエラーを記録します。これを完了済みと扱わないでください。

## 元データを別の場所へ移した場合

`--activation-manifest /new/path/manifest.json`、`--checkpoint /new/path/latest.pt`、
`--samples /new/path/norm-samples.jsonl`を指定できます。内容・split設定の整合性チェックは維持します。
