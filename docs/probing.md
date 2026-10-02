# Sparse probing：主目的の評価

Raw / ZCA baselineも同じcheckpoint指定で評価できます。作成・学習は
[baselines.md](baselines.md)を参照してください。比較表には前段名・λ・βを記録します。

`sj-probe`は既存のstage-2 checkpointを固定し、SAE特徴のTop-1 / Top-2 / Top-5で
二値分類するprobeを学習します。SAEやLLMの再学習は行いません。
**主指標はTop-1 accuracy**です。SAE内部のK=64とprobeが使う特徴数は別です。
再構成FVU、非発火率、Gaussian指標は補助指標です。

## プロトコル

1. 明示したtrain / validation / testのラベル付きテキストを読みます。
2. 同一LLMの同一block出力を一度だけ収集し、全λで共有します。
3. 各トークンで固定encoder→stage-2 calibration→Top-K SAEを適用してから、
   padding・特殊トークン・前段のburn-inを除いて**SAE特徴を平均pool**します。
   活性を平均してからSAEに入れる方法とは異なります。
4. trainの正負クラスの平均特徴値の差の絶対値で特徴を選びます（同点はindex順）。
5. 選んだ特徴のみでL2 logistic regressionをCPU float64 / LBFGSで学習します。
   C候補の既定は0.1, 1, 10。validation accuracyで選び、同点では小さいCを採用します。
   testを特徴選択・C選択・早期停止に使いません。
6. dataset内のtask平均と、dataset間の等重み平均を記録します。
   タスク数の違いを確認できるよう、全task等重み平均も別に保存します。

データ準備と平均差による選択は
[SAEBenchの実装](https://github.com/adamkarvonen/SAEBench/blob/main/sae_bench/evals/sparse_probing/probe_training.py)
を参照しています。ただしvalidation分割、重複除去、solver、トークン処理、モデルが異なるため、
**論文Table 2や公式SAEBenchスコアの厳密再現ではありません**。
現在の目的は、同一の評価条件でλ=0 / 0.0003 / 0.001を比較することです。

## 実行例

```bash
pip install -e '.[probe]'
# 以下は任意のSAEBenchデータ準備用。独自のJSONLなら不要。
pip install 'git+https://github.com/adamkarvonen/SAEBench.git'

# まずAG Newsの複数二値タスクでpilot。
sj-probe prepare --datasets fancyzhx/ag_news \
  --train-size 4000 --test-size 1000 --seed 42 \
  --output data/probe/ag-news.jsonl

# Pythia-6.9Bをロードし、checkpointが指定するlayerの活性を一度だけ収集。
sj-probe collect --tasks data/probe/ag-news.jsonl \
  --checkpoint runs/stage2-pilot/model-00/checkpoints/latest.pt \
  --output data/probe/ag-news-activations --device cuda --batch-size 8

sj-probe evaluate --tasks data/probe/ag-news.jsonl \
  --activations data/probe/ag-news-activations \
  --checkpoints \
    runs/stage2-pilot/model-00/checkpoints/latest.pt \
    runs/stage2-pilot/model-01/checkpoints/latest.pt \
    runs/stage2-pilot/model-02/checkpoints/latest.pt \
  --output runs/probe-ag-news --device cuda
```

`prepare`はsource trainからvalidationを20%確保し、クラスごとに他クラスから負例を取り、
正負を同数にします。サイズは上流ローダへ渡す引数で、重複除去後の件数は減る場合があります。
`--datasets`を省略すると、SAEBenchのbios / reviews / code / news / europarlの8系列を使います。
データ取得の可否はHugging Faceと上流ローダに依存します。モデルやデータの取得にはネット接続が必要です。
prepareが書くprovenanceと元JSONLを保存してください。

task JSONLを準備した後は、収集と3条件の評価をまとめて実行することもできます。

```bash
PROBE_TASKS=data/probe/ag-news.jsonl \
ACTIVATION_CACHE=data/probe/ag-news-activations \
STAGE2_ROOT=runs/stage2-pilot RUN_ROOT=runs/probe-ag-news \
bash scripts/probe_sweep.sh
```

`collect`の既定context lengthは128、dtypeはbfloat16です。no automatic BOS/EOSで抽出します。
モデル名・revision・layer・hook・入力次元・先頭除外がcheckpointと一致しないと停止します。
元checkpointのrevisionが`main`のような可変名なら、抽出時の重みとの完全一致は証明できません。
実際に取得したmodel commitとtokenizer語彙hashをcache manifestへ記録します。
長文の切り詰め後に同一token列がsplitをまたぐ場合も停止します。

活性cacheはGB〜数十GB以上になることがあります。LLMをGPUに載せられないときは
batch sizeを減らしてもモデル本体のメモリは減りません。既存LLM環境で実行してください。
cacheはchecksum検証後に再利用できます。不完全cacheは上書きせず、別の出力先を指定してください。

## 独自データ

1行1例のJSONLで、必須フィールドは次の通りです。

```json
{"id":"doc-1","group":"doc-1","task":"topic/science","dataset":"custom","split":"train","label":1,"text":"..."}
```

各taskの各splitに0と1の両方が必要です。同一task内でid重複、同一文書groupのsplit間共有、
Unicode正規化・空白正規化後の重複テキストを拒否します。
同じ文書由来の複数例には共通のgroupを指定してください。
既存The Pile活性だけでは意味ラベルがないためprobeを計算できません。

## 出力・最終test評価

- `summary.md`：Top-1を主指標とした比較表。
- `model-XX.json`：各task・各kのaccuracy、balanced accuracy、選択特徴、係数、C候補のvalidation成績、
  最適化のiteration数と最終勾配。上限到達や大きな勾配がある場合は収束を点検してください。
- `comparison.json`：task/cache/checkpointのhash、引数、集約スコア。

既定はvalidationのみ報告します。条件を決めた後に`evaluate`へ`--include-test`を付け、
別の出力先で最終testを評価します。各probeの学習・C選択は元と同じtrain/validationで行い、
train+validationで再学習はしません。test結果を見て再調整すると、そのtestは最終確認用ではなくなります。
`--standardize`を指定した場合のみ、選択後の特徴をtrain平均・標準偏差で標準化します。
既定はSAEBenchの疎probeに合わせて標準化なしです。

この評価は意味概念の線形な取り出しやすさを測ります。accuracyだけで因果的忠実性や
あらゆる意味での解釈可能性を証明するものではありません。まずpilot、次にタスクを増やし、
有望な条件で学習seedを増やしてください。
