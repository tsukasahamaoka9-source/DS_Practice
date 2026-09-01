# 2026-09-02 Kaggle日次レビュー

今日の3本を貫くテーマは **「指標の構造を読んで、設計に落とす」**。

- Biohub は「LBを使わずに公式指標を自前で再現し、変更の良し悪しを測る」。
- Playground S6E9 は「AUCは順位しか見ない」という性質を、ブレンドの設計に丸ごと利用する。
- RSNA Knee は「macro平均AUC = 12個の独立問題」という構造から、ターゲット別の重み・ゲート・キャリブレーションを導く。

3本とも、手法そのものより **「なぜその設計になるのかを指標から説明できる」** ことが共通の強みです。

---

## 1. Biohub - Cell Tracking During Development(固定枠)

| 項目 | 内容 |
|---|---|
| notebook | [Biohub 0.934 LB PROXY_SCORE=0.9384](https://www.kaggle.com/code/evgendvorkin/biohub-0-934-lb-proxy-score-0-9384) |
| 原著者 | ДВОРКИН ЕВГЕНИЙ ВЛАДИМИРОВИЧ (evgendvorkin) |
| Public Score | **0.934**(V27 / 85 votes・Gold / GPU T4 x2 で 31分6秒) |
| ローカルファイル | `biohub_biohub-0-934-lb-proxy-score-0-9384.ipynb` |

### 学べる主要テクニック

1. **公式指標のローカル再実装(PROXY_SCORE)** — train動画からテストに含まれないものを選び、ハンガリアン法によるノード照合(7µmゲート)・Edge Jaccard・ノード数ペナルティ付きAdjusted Jaccard・Division Jaccard をすべて自前で実装。LBの提出枠を消費せずに変更を評価する。
2. **イベント認識のバリデーション標本設計** — 分裂がGTに注釈されている動画は**約44%しかない**ため、候補ごとにGTグラフを開いて「出次数2以上のノードがあるか」を実際に確認し、分裂を含む動画を優先して選ぶ。**測りたい現象を含まない検証セットを構造的に排除する**。
3. **Configuration Guard** — 重要パラメータ8個を期待値表と照合し、1つでも違えば `RuntimeError`。「実行できた = 設定は意図どおり」を保証する10行。
4. **Pipeline Manifest** — 設定値ではなく**実際にロードされたか**を `Path.exists()` と `globals()` で観測して印字。`requested=True, loaded=False` を明示的に警告する。
5. **調和平均による双方向エッジ融合** — 順方向 p_fwd と逆方向 p_bwd の調和平均 `2·p_fwd·p_bwd/(p_fwd+p_bwd)` は小さい方に強く引きずられるので、片方向だけが支持するFPエッジを潰す。
6. **frame retention guard** — 融合後の候補数が primary の90%を下回ったフレームは丸ごと primary に巻き戻し、判断をJSONLに記録。**アンサンブルの下振れだけを止める**。

### 評価指標の要約

Edge Jaccard(約85%)+ Division Jaccard(約15%)の重み付き和。Jaccard は `TP/(TP+FP+FN)` なので**過剰検出と見落としを同時に罰する**。座標照合は**µm物理空間**で行われ、`z=1.625, y=x=0.40625 µm/voxel` なので **XY重心の精度はZの4倍重い**。ノード数が `estimated_number_of_nodes` を超えると減点。

### 改善点の考察

**他notebookとの比較**(同コンペ Code タブ Best Score 上位を確認):

| notebook | Score | このnotebookに無い要素 |
|---|---|---|
| [Biohub Solution](https://www.kaggle.com/code/kaiwalyaatulraut/biohub-solution) | 0.966 | 0.93帯を大きく超える別系統。ただし votes 48・Bronze で再現性の検証が薄い |
| [Biohub Competition Solution](https://www.kaggle.com/code/kaiwalyaatulraut/biohub-competition-solution) | 0.965 | 同上 |
| [Biohub - Track Your Cells Development](https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells-development) | 0.933 | 同スコア帯だが別アプローチ。**3seed以上のアンサンブル** |
| [Biohub Cell Tracking](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking) | 0.926 | — |

0.965〜0.966帯が公開されている一方、このnotebookは0.934。**約0.03の差がどこから来るのかが最大の論点**です。

**関連文献**:

- **HOCT (Higher-Order Cell Tracking Transformer, arXiv 2607.11754)** — 候補追跡グラフの**エッジ中心**アーキテクチャ。「分裂が系譜経路をノード埋め込み空間で絡ませる」「ノードを共有するエッジのラベル一致がほぼランダムなので、候補グラフのトポロジがGNNの集約にとって無情報になる」という2つの構造的障害を指摘し、**エッジ同士を3D幾何事前分布の下で相互にattendさせる**ことで解決。深い事前学習画像エンコーダなしで Cell Tracking Challenge SOTA。このnotebookの bidirectional fusion は「1本のエッジについて2方向を見る」に留まっており、**エッジ間の相互作用は見ていない**。
- **Trackastra / Cell-TRACTR** — 手調整のリンキングを Transformer に置き換え、分裂も含めて end-to-end で学習する流れ。このnotebookの後処理は依然としてしきい値の手調整が支配的。
- **Cell-HOTA** — 分裂の**早すぎ/遅すぎ**を区別して評価する指標。コンペのDivision Jaccardは二値判定なので、「1フレームずれた分裂」が丸ごとFPになっている可能性がある。

**改善提案(5点)**:

1. **PROXY_SCORE を「差分の統計的有意性」まで含めて使う。** 現状は proxy 0.9384 に対し LB 0.934 と約0.004の楽観バイアスがあり、**このバイアスと同程度の改善は誤差と区別がつきません**。検証動画数(タイプごと2本)を増やすか、動画単位のブートストラップで Δproxy の信頼区間を出すべき。「上がった/下がった」ではなく「有意に上がったか」で判断できるようになります。
2. **`decompose_errors` の出力を目的関数に接続する。** すでに失点を「見逃し/余剰/誤接続」に分解できているのに、パラメータ探索は手動です。「FN由来の失点が支配的なら `DISAPPEARANCE_WEIGHT` を上げる」といったルールを自動化すれば、探索が仮説駆動になります。
3. **エッジ間の相互作用を導入する(HOCTの示唆)。** 現状の融合は各エッジ独立です。同じノードを共有するエッジ群を1つの集合として扱い、「この2本は排他」「この分岐は分裂と整合」といった高次の制約をスコアに反映させれば、ILPの前段でFP候補を減らせるはずです。ILPソルバの負荷も下がります。
4. **モンキーパッチの成否を assert する。** セル5は `read_text()` → `replace()` → `write_text()` で上流スクリプトを書き換えていますが、**置換対象の文字列が見つからなくても例外は出ません**。上流が更新されたら「8視点TTAのつもりで4視点で走る」という静かな劣化が起きます。`assert _old in _s, "TTA patch target not found"` を各パッチに入れるだけで防げます(Configuration Guard と同じ思想を、パッチ適用にも適用する)。
5. **分裂の時間的ズレを測る。** Cell-HOTA の指摘通り、分裂の検出位置が1フレームずれているだけでFP+FNになっている可能性があります。`decompose_errors` に「±1フレーム以内に対応するGT分裂があるか」の集計を足せば、`SAFE_DIV_*` のしきい値を触るべきか、**タイミング推定を直すべきか**を切り分けられます。0.965帯との0.03の差の一部はここかもしれません。

---

## 2. Playground Series S6E9(ローテーション枠)

| 項目 | 内容 |
|---|---|
| コンペ | [Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9)(9/1開始・9/30終了、255チーム) |
| notebook | [EV Adoption Tokens: XGBoost + TinyTokenTransformer](https://www.kaggle.com/code/chovyxu/ev-adoption-tokens-xgboost-tinytokentransformer) |
| 原著者 | CHOVY (chovyxu) |
| Best Score | **0.94539**(V2 / 17 votes・Bronze) |
| ローカルファイル | `playground_ev-adoption-tokens-xgboost-tinytokentransformer.ipynb` |

### 学べる主要テクニック

1. **「表の1行を小さな言語に翻訳する」トークン設計** — 数値フィールド1つ=1トークン、カテゴリ1つ=1トークン。トークンの中身に、ロバスト標準化値・欠損フラグ・**フーリエ基底 sin/cos(f=1,2,4,8)**・局所ガウス基底・分位ビン頻度・丸め値頻度を詰める。同じ情報から**XGBoost用の平坦行列**と**Transformer用のトークンテンソル**という2ビューを作る。
2. **リークの有無をクラス境界で物理的に分離** — `TokenFeatureState`(目的変数を使わない → train+testで fit してよい)と `TargetMeanEncoder`(目的変数を使う → fold内でしか fit してはいけない)を別クラスにする。**「この特徴はリークするか?」の判定基準は"目的変数を参照しているか"の1点だけ**。
3. **マルチスケール分位ビン target encoding** — 8/16/32/64ビンの4粒度で同時に作り、どれが効くかはモデルに選ばせる。
4. **forensic 特徴量** — 合成データの生成痕跡(丸め誤差、`value % 100` などの剰余、桁ごとの出現頻度)を拾う。
5. **OOF貪欲ブレンドを `prob` / `rank01` / `logit_rank01` の3ビューで回す** — AUCは順位不変なので、順位空間で混ぜてもスコアは損なわれず、**スケールの違うモデルを対等に足せる**。選択はAUC、同点時のタイブレークはlogloss。
6. **XGBoostのOOF予測をTransformerの入力トークンにする**(スタッキング)。OOFであることが絶対条件。

### 評価指標の要約

ROC AUC。**順位のみを見る指標**で、単調変換に対して完全に不変。有病率(正例率)に依存せず、「買いそうな順に並べて上位に販促する」という実務にも直結する。この notebook は選択基準をAUCにし、logloss/Brierを並走させて確率の質が壊れていないかを監視している。

### 改善点の考察

**他notebookとの比較**(S6E9 Codeタブ、開始2日目):

| notebook | Score | このnotebookに無い要素 |
|---|---|---|
| [S6E9 LightGBM](https://www.kaggle.com/code/kirill0212/s6e9-lightgbm) | **0.94607** | 現在の公開最高。ただし private notebook からのコピーで votes 4 |
| [Single XGB](https://www.kaggle.com/code/evgendvorkin/single-xgb) | 0.94327 | 単一モデル・27 votes。**シンプルな構成で0.943**という基準線 |
| [S6E9 starter: how to tell a real gain from noise](https://www.kaggle.com/code/georgymamarin/s6e9-starter-how-to-tell-a-real-gain-from-noise) | — | **LB分解能の実測**。改善が誤差かどうかの判定方法(21 votes) |
| [Reading a synthetic dataset: EV purchases (S6E9)](https://www.kaggle.com/code/tomasa2/reading-a-synthetic-dataset-ev-purchases-s6e9) | 0.94397 | 合成データの生成構造そのものの分析 |

注目すべきは **単一XGBの0.94327 と、この重厚なパイプラインの0.94539 の差が 0.002 しかない**ことです。フーリエ基底・Transformer・forensic特徴・3ビュー貪欲ブレンドを全部積み上げて +0.002。**費用対効果を正面から評価する必要があります**。

**関連文献**:

- **FT-Transformer (Gorishniy et al. 2021) と "On Embeddings for Numerical Features in Tabular Deep Learning" (arXiv 2203.05556)** — 「各数値・カテゴリ特徴をトークン化する」というこのnotebookの設計は FT-Transformer そのもの。論文の主張は、**数値特徴に高度な埋め込み(周期的埋め込み=まさにフーリエ基底、区分線形埋め込み)を与えると、深層表形式モデルとGBDTの差が埋まる**。この notebook の sin/cos 基底は論文の PLR 埋め込みの実装版です。
- **TabPFN v2 (Hollmann et al., Nature 637, 319–326)** — 約1億の合成データセットで事前学習した Transformer が、**10,000サンプル×500特徴までのデータで、4時間チューニングしたアンサンブルを2.8秒で上回る**。Playgroundのデータは行数がこれより多いですが、**サブサンプルしてTabPFNを1本足す**のはアンサンブルの多様性として非常に安価。
- **RuleNet / TabArena (2026)** — 学習可能なルール埋め込み + 区分線形分位射影 + 特徴マスキングアンサンブル。「特徴マスキングでアンサンブルを作る」という発想は、このnotebookの多ビュー戦略と補完的。

**改善提案(5点)**:

1. **積み上げた要素の ablation を取る(最優先)。** 単一XGB 0.94327 との差は0.002。5-foldのfold間標準偏差がどれくらいか分かりませんが、**forensic特徴・Transformer・スタックトークンをそれぞれ抜いたときのOOF AUCを測らないと、どれが効いているのか誰にも分かりません**。効いていない要素は、実行時間とLB過学習リスクを増やしているだけです。
2. **LBの分解能を先に測る。** 上の "how to tell a real gain from noise" が実践している通り、**public LBが何桁目まで信用できるか**(テスト行数から Hanley-McNeil のSEを見積もる、あるいは既知の2提出の差から実測する)を先に確認すべきです。分解能が0.0005なら、0.94327 → 0.94539 は「たぶん本物」、0.94539 → 0.94607 は「誤差の範囲」と判定できます。
3. **TabPFN v2 をアンサンブルに1本足す。** 帰納バイアスが GBDT とも Transformer とも違う(合成データ事前学習によるin-context学習)ので、**誤りが非相関になりやすい**。数万行にサブサンプルして複数回走らせて平均すれば、数分のコストで貪欲ブレンドの候補が1つ増えます。
4. **貪欲ブレンドの候補数を絞るか、ネストCVで検証する。** 現状は「4モデル × 3ビュー = 12候補」を OOF 上で貪欲探索しています。候補が増えるほど OOF に過学習し、**ブレンド後のOOF AUCは実力より楽観的**になります。ブレンド重み探索自体を fold の外に置く(ネストCV)か、候補を `rank01` に一本化してでも探索空間を狭める方が、private LB では安全です。
5. **forensic特徴の妥当性を明示的に検証する。** 剰余や丸め誤差は**合成データの生成器の癖を突いているだけ**で、汎化する知識ではありません。少なくとも「これらを入れた場合と外した場合のOOF差」を記録し、効果が小さければ外すべきです(次元数と学習時間の削減にもなる)。効果が大きい場合は、それ自体が「このデータの生成過程」についての知見なので、Discussionで共有する価値があります。

---

## 3. RSNA Knee Abnormality Detection(実コンペ枠)

| 項目 | 内容 |
|---|---|
| コンペ | [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)(2,857チーム、$77,000、10/22締切) |
| notebook | [RSNA Knee Abnormality DetectionV1](https://www.kaggle.com/code/kunaldesale2408/rsna-knee-abnormality-detectionv1) |
| 原著者 | KUNAL DESALE (kunaldesale2408) |
| Public Score | **0.936**(V5 / 62 votes・Silver / GPU T4 x2 で 18分38秒) |
| ローカルファイル | `competition_rsna-knee-abnormality-detectionv1.ipynb` |

### 学べる主要テクニック

1. **4アーム多様性設計** — DINOv2 ViT-S(自然画像・自己教師あり、0.899)/ Cross-Series Attention A5(DINOv3、0.910)/ RadImageNet ResNet-50(医療画像で事前学習、0.920)/ CoAtNet Raptor(64スライス体積、0.936)。**事前学習の分布・入力の作り方・解像度が全部違う**ので誤りが非相関。
2. **スロット化(slot filling)** — シリーズ本数もスライス数も検査ごとに違う入力を、「撮像面 × パルスシーケンス」で定義した固定枠に埋める。無いスロットはゼロ埋め + マスク。
3. **解剖学的事前分布のハードコード** — `SLOT_PRIOR_TABLE` で所見ごとに見るべきスロットを制限(ACLは矢状断、MCLは冠状断)。ラベルが実質58件しかないので、**ドメイン知識が正則化として働く**。
4. **プロトコル対応キャリブレーション** — 88個のメタデータ由来特徴で施設・装置間のバイアスを線形補正。20か国以上のデータなので、施設バイアスは直接AUCを下げる。
5. **LLM由来のソフトラベル** — 4,349件のフリーテキスト読影レポートを言語モデルに読ませ、12個の**確率**に変換して教師信号にする。「疑われる」は1.0ではなく0.8。本物のラベル58件は学習に使わず**検証専用**(そこでのmacro-AUC 0.9167)。
6. **相関ゲート型のターゲット別ランク融合** — 相関 > 0.992 なら多様性が無いので保守化、< 0.60 ならどちらか壊れている疑いで保守化。コメント曰く *"Correlation only controls risk. It never creates a new target weight."*
7. **フェイルセーフ** — 最終融合を try/except/finally で囲み、失敗したら検証済みの Stage 3 提出に自動で巻き戻す。

### 評価指標の要約

**macro平均 AUC**:`(1/12)·Σ AUC_i`。12所見の有病率が桁違いに違うため、micro平均だと頻度の高い所見だけで点数が決まってしまう。macro平均は各所見に等しく1/12を与えるので、稀だが臨床的に重要な所見(骨折など)が正当に評価される。

**この構造から導かれる決定的な帰結**: macro平均の最大化は**12個の独立した最適化問題**と同値。ターゲット間にトレードオフが無いので、**ターゲット別に融合重み・ゲート・キャリブレーションを変えてよい(むしろ変えるべき)**。全ターゲット共通の重み1つを使うのは明確に損。

**Efficiency Prize トラック**(賞金 $18,000、今週 Efficiency LB が公開):

```
Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400   ← 最小化
```

精度と実行時間の両方で評価される第2トラックです。この4アーム構成は18分38秒(1,118秒)で `1118/32400 ≈ 0.0345` の実行時間ペナルティ。

### 改善点の考察

**他notebookとの比較**(Codeタブ Best Score 上位):

| notebook | Score | 特徴 |
|---|---|---|
| [RSNA Knee \| DINOsaur V4 🦖](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4) | 0.936 | 同スコア。Stage 4 の原型 |
| [RSNA Knee: Take Care Of Your Knee](https://www.kaggle.com/code/anhadmahajan06/rsna-knee-take-care-of-your-knee) | 0.936 | 同スコア・別系統 |
| [RSNA Knee: read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | 0.906 | **レポートテキストを推論時にも使う**(このnotebookは画像のみ) |
| [RSNA Knee Abnormalities - Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | — | 主催者の効率トラックLB(305 votes・21,041 views) |
| [RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 0.891 | 484 votes。単一アームの基準線 |

**公開最高帯が 0.936 で3本以上が横並び**しており、天井に張り付いている状態です。ここから抜けるには、アームを足す方向ではなく**別の情報源**が要ります。

**関連文献**:

- **eClinicalMedicine (2025) 9所見マルチタスク膝MRI** — 大規模・多施設・段階的検証で9つの膝異常を分類。放射線科医と併用して読影精度が向上。**マルチタスク学習が単一タスクの寄せ集めより有利**という報告は、12個を別々に扱う現行設計への対案になります。
- **European Radiology (2025) 23条件のマルチ組織解析** — **3D slice transformer** で軟骨・半月板・骨髄・靭帯を横断的に扱う。スライスを2Dに落とさず3Dのまま扱う方向。
- **"Comparative Evaluation of Deep Learning and Foundation Model Embeddings for Osteoarthritis Feature Classification" (2025)** — 44,985枚の膝X線で、教師あり CNN(ResNet18, ConvNeXt-Small)と **BiomedCLIP / RAD-DINO** の基盤モデル埋め込みを比較。**RAD-DINO はこのnotebookが使っていない医療特化の自己教師ありViT**で、DINOv2/v3 の自然画像版より膝MRIに適合する可能性が高い。
- **OrthoDiffusion (arXiv 2602.20752)** — 筋骨格MRI向けの汎用マルチタスク拡散基盤モデル。

**改善提案(5点)**:

1. **読影レポートを推論時にも使う(最有力)。** このコンペの最大の特徴は「全スタディにレポートが付いている」ことなのに、この notebook はレポートを**学習ラベルの生成にしか使っていません**。0.906 の "read the report, then the knee" が示す通り、テキストは推論時にも情報を持ちます。テキストアームを1本足せば、**画像アーム同士より遥かに誤りが非相関**なはずです。macro平均AUCなので、テキストが強い所見(Synovitis など記述が定型的なもの)だけ重みを上げればよい。
2. **RAD-DINO / BiomedCLIP を5本目のアームに。** 現行4アームのうち医療画像で事前学習しているのは RadImageNet だけです。RAD-DINO は医療画像の自己教師ありViTで、**DINOv2/v3 と同じアーキテクチャ・違う事前学習分布**という理想的な多様性源になります。差し替えではなく追加で。
3. **consensus power calibration(γ=1.05)を外して検証する。** この変換は**単調なので、理論上 AUC を1ミリも動かしません**(float丸めで同着が分離される程度)。にもかかわらず焦点4ターゲットにだけ適用されているのは、**LBへの過学習の痕跡**である可能性が高い。外して同スコアなら、パイプラインが1段シンプルになります。もし本当に効かせたいなら、**融合の前**にかけるべき(混合比が実質的に変わるため)。
4. **アーム間のグローバル名前空間汚染を解消する。** `_A5_SAVED = dict(globals())` で退避しているのは対症療法で、セル1の `SLOTS`(6要素タプル)とセル2の `SLOTS`(2要素タプル)は構造すら違います。**各アームを関数かクラスに閉じ込める**だけで、実行順依存の事故が消え、アームの追加・削除も安全になります。0.936帯が横並びの今、次に効くのは新手法よりも**実験を安全に回せる基盤**です。
5. **Efficiency トラックを別提出として狙う。** 現行は4アーム18分38秒。`Efficiency = AUC/(Benchmark−maxAUC) + Runtime/32400` を見ると、**AUCをわずかに落としてでも実行時間を大幅に削る方が有利になる領域**があります。単体0.920 の RadImageNet アーム1本(推論時間はおそらく1/4以下)で提出すれば、メインLBでは負けても効率トラックでは上位に入る可能性があります。**賞金$18,000の枠に、既存の資産だけで挑戦できる**のは費用対効果が非常に高い。

---

## 今日のまとめ

3本に共通するのは **「指標を読んで設計を決める」** という一点です。

- Biohub: Jaccard が FP と FN の両方を罰する → 融合はFPを削る方向(調和平均)、後処理はFNを増やさない方向(retention guard)。そして**評価そのものを手元で再現**して、LBに頼らず判断する。
- S6E9: AUC は順位不変 → **順位空間でブレンドしても損しない**。だからスケールの違うモデルを対等に足せる。
- RSNA: macro平均 = 12個の独立問題 → **ターゲットごとに設計を変えてよい**。

そして3本とも、**「効いたと言うために何を測るか」を先に決めている**点で一致しています。Biohub の PROXY_SCORE、S6E9 の OOF + logloss タイブレーク、RSNA の58件の本物ラベルによる検証。いずれも「LBの数字」以外の判断軸を持っています。

一方で改善提案の第1位が3本とも **「積み上げた要素の効果を個別に測れ」** になったのは偶然ではありません。要素を足すのは簡単で、効いているかを測るのは面倒 ―― この非対称性が、公開notebookが天井に張り付く一番の理由なのかもしれません。

---

## 参考文献・出典

- [Higher-Order Cell Tracking Transformer (arXiv 2607.11754)](https://arxiv.org/abs/2607.11754)
- [Cell-TRACTR: A transformer-based model for end-to-end segmentation and tracking of cells (PLOS Comput Biol)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013071)
- [On Embeddings for Numerical Features in Tabular Deep Learning (arXiv 2203.05556)](https://arxiv.org/pdf/2203.05556)
- [Revisiting Deep Learning Models for Tabular Data (FT-Transformer)](https://openreview.net/pdf?id=i_Q1yrOegLY)
- [Tabular Machine Learning 2026: TabArena Leaderboard — TabPFN vs XGBoost](https://www.codesota.com/tasks/tabular-ml)
- [Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI (eClinicalMedicine)](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(25)00467-5/fulltext)
- [Comprehensive deep learning-assisted multi-condition analysis of knee MRI studies (European Radiology)](https://link.springer.com/article/10.1007/s00330-025-12052-8)
- [Comparative Evaluation of Deep Learning and Foundation Model Embeddings for Osteoarthritis Feature Classification](https://link.springer.com/article/10.1007/s10278-025-01636-x)
- [OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI (arXiv 2602.20752)](https://arxiv.org/pdf/2602.20752)
