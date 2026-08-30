# 2026-08-31 Kaggle日次レビュー

本日扱った3本。いずれも**学習目的の解説付き写し**で、原著コードは変更せず、各コードセルの直前に日本語のWhat/Why解説Markdownセルを挿入している（未実行のため出力は含まない）。

| 枠 | コンペ | notebook | 原著者 | スコア |
|---|---|---|---|---|
| 固定枠 | Biohub - Cell Tracking During Development | [Biohub 0.933 LB](https://www.kaggle.com/code/evgendvorkin/biohub-0-933-lb) | evgendvorkin | Public LB **0.933** |
| Playground | Predicting Smartphone Addiction (S6E8) | [s6e8 smartphone addiction eda fast](https://www.kaggle.com/code/lamhuy8904/s6e8-smartphone-addiction-eda-fast) | lamhuy8904 (LÂM HUY) | Public LB **0.97125** |
| 実コンペ | RSNA Knee Abnormality Detection | [RSNA Baseline](https://www.kaggle.com/code/evgendvorkin/rsna-baseline) | evgendvorkin | Public LB **0.936** |

---

## 1. Biohub - Cell Tracking During Development

**ファイル**: `biohub_biohub-0-933-lb.ipynb`

### 学べる主要テクニック

- **Dual-seedアンサンブル + 調和平均による双方向エッジ融合**: 乱数シードの異なる TemporalUNet3D 2本の検出ロジット・エッジロジットを重み 0.85 で混合し、順方向/逆方向の関連確率を harmonic probability で統合する。
- **frame retention guard（フレーム単位のロールバック）**: アンサンブル後の候補ノード数が主モデル単独の90%を下回ったフレームは、主モデルの結果に差し戻す。「アンサンブルは平均的には良いが、局所的には悪化しうる」という現実への安全弁で、v2→v4の改善（+0.0035）の主因。
- **モンキーパッチによる上流スクリプト改変**: 外部リポジトリの `predict_unet_transformer.py` をソース文字列置換で書き換え、8方向TTA・secondaryモデル・guardを注入する。置換文字列が見つからなければ即エラーにする防御付き。
- **密度適応型のgap closing閾値**: 細胞が密集した領域では接続閾値（6.5µm）を自動的に縮め、誤接続を抑えつつ疎な領域では積極的に切れたトラックを繋ぐ。
- **Config-Driftガードとパイプライン・マニフェスト**: 実行前に環境変数が期待値と一致するかを検証し、実行後に「設定上ONの機構が実際に有効化されたか」を印字する。31分の推論を無駄にしないための投資。
- **label-free監査**: 正解を使わず、スキーマ・ID連番・グラフのトポロジー（1親→3子がないか等）だけで submission の妥当性を検証する。

### 評価指標の要約

ノードをハンガリアン法で7µm以内に対応付けたうえで計算する **Adjusted Edge Jaccard**（ノード数超過にペナルティ）を主軸に、稀少イベントである **Division Jaccard** を重み0.1程度で加味した合成スコア。エッジ（細胞のつながり）を直接評価しつつ、「ノードを撒き散らして偶然当てる」ハックをノード数ペナルティで封じる設計になっている。

### 改善点の考察

**他notebookとの比較**（Codeタブ Best Score上位を確認）:
同コンペには `kaiwalyaatulraut/biohub-solution`（0.966）や `kaiwalyaatulraut/biohub-competition-solution`（0.965）といった、本notebookを大きく上回るスコアが存在する。ただしこれらは過去のレビュー（2026-08-12 / 08-13）で確認した通り**メトリックハック寄りの手法**（提出フォーマットの隙を突いたダミーノード/エッジのパディング等）を含んでおり、学習素材としての価値は低い。一方、本notebookが**採用していない正攻法の手法**として以下がある:

- `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery`（0.894）が持つ**学習ベースのgap recovery**（距離ルールではなく、復活候補を学習済み分類器でスコアリングする）は、本notebookの距離閾値ベースのgap closingより表現力が高い。本notebookは密度適応を入れているが、学習ベースの再スコアリングは入れていない。
- 3シード以上への拡張。現状 dual-seed（2本）だが、`SECONDARY_DETECTION_WEIGHT` を単一のスカラーで固定しているため、3本目を足すには重み設計を作り直す必要がある。

**関連文献**:
[Higher-Order Cell Tracking Transformer (HOCT)](https://arxiv.org/html/2607.11754) は、候補リンク（エッジ）同士を3D幾何事前分布のもとで相互にattentionさせるエッジ中心アーキテクチャで、**深層の事前学習済み画像エンコーダなしで** Cell Tracking Challenge のSOTAを達成している。本notebookのNode Transformerがペア単位で独立に判定しているのに対し、HOCTは「このエッジを採用するなら隣のエッジは採用しにくい」という競合関係をモデル内部で扱える。また [TRACKASTRA](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_27) は、ILPソルバーを使わずtransformerだけでリンクを解く方向性を示している。

**改善提案**:

1. **gap closing を学習ベースに置き換える**。現状は距離6.5µm＋密度補正という手作りルール。切れたトラックの終端・始端ペアに対し、距離・輝度・移動方向の一致度・周辺密度などを特徴にした軽量分類器（LightGBM等）で「繋ぐべきか」をスコアリングし、その確率をILPのエッジ重みに入れる。ルールでは表現できない相互作用を拾える。
2. **エッジ間の競合をILPの外でも扱う**。HOCT的なエッジ中心attentionを Node Transformer の後段に足し、「同じノードを取り合う複数のエッジ」を同時に見てからILPに渡す。現状はILPが制約だけでこれを処理しており、確率の質そのものは競合を考慮していない。
3. **frame retention guard の閾値を検証で探索する**。現在 `DUAL_SEED_MIN_CANDIDATE_RETENTION = 0.90` は固定値。この値はvalidationで最適化された形跡がなく、0.85や0.95で挙動が変わる可能性が高い。guard発動回数とPROXY_SCOREの関係をプロットするだけで判断材料になる。
4. **Division Jaccard への投資**。validation実測で Division Jaccard = 0.1667（Division FP = 0）と、**分裂検出はほぼ機能していない**（FPは0だがTPも極端に少ない = 保守的すぎる）。合成スコアでの重みが小さいため放置されているが、`SAFE_DIV` の距離帯（7.0〜12.0µm）を緩めてFPを許容する方向で、Division Jaccardの伸び幅とEdge Jaccardの劣化幅のトレードオフを測る価値がある。
5. **validationの分散を測る**。`VALIDATOR_N_PER_TYPE = 2` は動画数が少なく、PROXY 0.9396 と 0.9361 の差が有意かどうか判断できない。n を増やすか、動画単位のbootstrapで信頼区間を出すべき。

### まとめ

「アンサンブルで局所的に悪化しうる」という現実に対し **frame retention guard** という具体的な安全弁を用意した点が本notebookの独自性。設定ガード・整合性チェック・label-free監査・パイプライン・マニフェストと、**長時間パイプラインを壊さないための防御的設計が一貫している**のが最大の学びどころ。

---

## 2. Predicting Smartphone Addiction (Playground Series S6E8)

**ファイル**: `playground_s6e8-smartphone-addiction-eda-fast.ipynb`

### 学べる主要テクニック

- **Out-of-Fold ベイジアン平滑化ターゲットエンコーディング**: `(カテゴリ内合計 + 全体平均 × 20) / (件数 + 20)` の縮小推定を、foldごとに学習側インデックスだけから計算。リークと少数カテゴリの暴れを同時に潰す定番実装。
- **比率・残差による特徴量エンジニアリング**: `screen_to_waking_ratio`、`unaccounted_screen_time`（スクリーン時間 − SNS − ゲーム − 仕事）、`opens_per_screen_hour` など。GBDTは `a/b` を軸平行分割で近似するのが苦手なので、人間が先に割っておくと1分割で使える。
- **小数桁特徴（Decimal Lattice）**: `frac_x = x - floor(x)` と `d1_x = floor(x*10) % 10`。合成データ生成器（CTGAN）が残した量子化の痕跡を拾う。
- **ロジット → パーセンタイルランク変換 → SLSQP による重み最適化**: AUCは順位しか見ないので、スケールの異なる3モデルをランクに揃えてから混ぜる。目的関数は代理損失ではなく `-roc_auc_score` そのもの。
- **transductive 頻度エンコーディング**: train+testを連結した全体での値の出現頻度。

### 評価指標の要約

**ROC AUC**。「ランダムに選んだ正例が、ランダムに選んだ負例より高いスコアを持つ確率」に等しく、**順位だけを評価しキャリブレーションは問わない**。クラス不均衡に頑健なため、合成データのPlaygroundで標準的に採用される。本notebookのランク変換ブレンドと重み直接最適化は、この性質を正面から利用した設計。

### 改善点の考察

> ⚠️ **最も重要な観察**: このnotebookの最終予測は `0.725 × 外部アンカー1 + 0.225 × 外部アンカー2 + 0.050 × 自前ブレンド` である。すなわち **LB 0.97125 のうち自前モデルの寄与はわずか5%** で、95%はアタッチした他人の公開submission CSVに由来する。さらに2件のIDを0/1に決め打ちする "Duplicate Magic" が入っている。前半のFEとCVは技術的に正当で学ぶ価値があるが、**スコアの数字はこの手法の実力を表していない**。

**他notebookとの比較**:

- `najiama/pure-lgbm-model-lb-0-96999-cv-0-96881`（51 votes）は**外部アンカーを使わない単一LightGBM**で LB 0.96999 / CV 0.96881 を出している。CVとLBの乖離が0.0012しかなく、これが「自前の実力」の目安。0.97125との差 0.0013 の大半が公開ブレンド由来だと推定できる。
- `raykkretzschmar/why-every-s6e8-notebook-above-0-97110-overfits`（0.97115）と `georgymamarin/s6e8-will-your-0-971-survive-the-private-split`（Silver, 50 votes）は、**上位帯が全員同じ公開予測に収束していること自体がprivate shakeupのリスク**であると論じている。本コンペは本日が最終日であり、まさにその検証が行われるタイミング。
- `koushikkumardinda/robust-ensembling-target-encoding-pipeline`（0.96479）は、本notebookが採用していない**ネストCVによるターゲットエンコーディングの検証**を行っている。本notebookのOOF TEはfold内では正しいが、平滑化パラメータ `SMOOTHING=20` を検証せずに固定している。

**関連文献**:
[Position: AI Competitions Provide the Gold Standard for Empirical Rigor in GenAI Evaluation](https://arxiv.org/pdf/2505.00612) は、10年以上のKaggleコンペで多様なリーク問題が特定されてきた一方、**オープンな challenge benchmark では参加者が優位に繋がるものを何でも突くため、リークのリスクが増幅される**と指摘している。また [On Privacy Leakage in Tabular Diffusion Models](https://arxiv.org/html/2605.06835v1) では、合成データの「有用性が高いほど元データへの過学習（＝痕跡の残存）が起きやすい」というトレードオフが定量化されており、Playgroundの小数格子特徴が効く理由の理論的背景になっている。

**改善提案**:

1. **アンカー抜きの自前スコアを必ず測る**。`sota_anchors = []` を強制して実行し、自前パイプライン単独のLBを確認する。これをやらない限り、FEやモデル選択の改善が本物か判定できない。おそらく 0.9698〜0.9702 程度に着地する。
2. **`SMOOTHING` をネストCVで探索する**。現在20固定。カテゴリの件数分布（`age` は種類が多く、`gender` は3種類）によって最適値は大きく違うはずで、列ごとに変えるだけで単体スコアが動く可能性がある。
3. **小数格子特徴の寄与を条件付きで反証する**。`frac_*` / `d1_*` を全部落としたときのCV低下幅を測る。もし大きく下がるなら、そのスコアは「データ生成器の癖の暗記」であり、private splitでも同じ癖が残っている保証はない（同一生成器なら残るが、Kaggleが分割方法を変えていれば消える）。
4. **Duplicate Magic を削除する**。AUCへの寄与はサンプル2件分でほぼゼロなのに、コードの再現性と可搬性を落としている。学習素材としては有害。
5. **多様性を rank correlation で定量する**。3モデル（LGB/XGB/CB）のOOFランク相関を出し、0.99を超えているなら実質1モデルであり、重み最適化に意味はない。多様性を作るなら、seed違いではなく**特徴量サブセットや目的関数（rank:pairwise 等）を変える**べき。

### まとめ

前半（FE + OOF Target Encoding + ランク融合）は表形式コンペの教科書的な良い実装。後半の public blend と Duplicate Magic は、**「高スコアnotebookのスコアがどこから来ているかを分解して読む」練習台**として価値がある。数字を見てから手法を評価するのではなく、寄与を分解してから評価する習慣をつけたい。

---

## 3. RSNA Knee Abnormality Detection

**ファイル**: `competition_rsna-baseline.ipynb`

### 学べる主要テクニック

- **LLMによるレポート→ソフトラベル生成（弱教師あり学習）**: 公式正解は58研究しかないが、放射線レポートの自由記述をLLMで12所見の**確率**（0/1ではない）に変換し、4,349研究の学習データを獲得している。「軽度の変性が疑われる」といった確信度の差をそのままラベルに残せる。
- **アーキテクチャ×事前学習ドメインの多様化による4ブランチ構成**: DINOv2（自然画像・自己教師あり・局所3スライス）/ DINOv3（同・16スライスの疑似3D）/ RadImageNet ResNet50（医用画像事前学習・CNN）/ CoAtNet（CNN+Transformerハイブリッド・64スライス）。誤りの傾向が構造的に違う4本を混ぜている。
- **所見ごとのSlotHead attention**: 「ACLは矢状断、Baker嚢腫は軸位断」という臨床知識を、attentionで自動獲得させる。macro AUCでは稀少所見1つの改善が全体の1/12を動かすため直接効く。
- **物理単位（mm）でのクロップと解剖学的スライス帯選択**: `CROP_MM = 130.0`、`SLICE_BAND = (0.12, 0.88)`。DICOMのpixel spacingを読んで実寸で切ることで、患者・装置間の写る範囲のばらつきを吸収する。
- **ランク平均 + 貪欲前向き選択**: 候補モデルを1つずつ足してmacro AUCが上がるものだけ残す。

### 評価指標の要約

**macro-averaged ROC AUC**（12所見それぞれのAUCの単純平均）。所見ごとの有病率が桁違い（変形性関節症は頻出、骨折やBaker嚢腫は稀）なため、microだと稀少所見を全部落としても高得点になってしまう。macroは12所見に同じ重みを与え、かつ順位のみを見るので所見ごとの閾値設計を運用側に委ねられる。この構造が SlotHead（所見別の注意配分）とランク平均アンサンブルを合理化している。

### 改善点の考察

**他notebookとの比較**:

- `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`（0.924, 96 votes）は、**単一モデルで0.924**を出している。本notebookの4ブランチ・30モデル超のアンサンブルとの差は 0.012 しかない。推論時間・複雑さのコストを考えると、**アンサンブルの費用対効果は低い**。実際、同コンペには `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`（292 votes, Kaggle公式）という効率トラックのリーダーボードが用意されており、精度だけを追う設計は評価軸を1つ落としている。
- `prvsiyan/head-and-shoulders-knees-and-toes`（0.936, 147 votes）は同スコアだが、より整理された構成で解剖学的整合ゲートを持つ。本notebookが採用していない要素として、**所見間の共起事前知識による補正**（例: 内側半月板損傷とMCL損傷は共起しやすい）がある。本notebookは12所見を完全に独立に扱っている。
- `tonylica/rsna-knee-dino-radimagenet-rank-ensemble`（0.92）と比べると、本notebookはCoAtNetブランチの追加が差分。ただしCoAtNetは64スライス×384pxと最も重く、寄与の割に推論コストが大きい可能性がある。

**関連文献**:
[Large Language Model-Based Uncertainty-Adjusted Label Extraction for AI Model Development in Upper Extremity Radiography](https://arxiv.org/pdf/2510.05664) は、本notebookとまさに同じ「LLMでレポートから不確実性込みのラベルを抽出する」枠組みを扱っており、**不確実性を明示的に調整したラベル**が下流の分類性能を改善することを報告している。また [多読者・多モデルのベンチマーク（Skeletal Radiology 2026）](https://link.springer.com/article/10.1007/s00256-026-05342-9) では、膝MRIレポートからのOuterbridge軟骨グレード抽出を7つのLLMで比較しており、**モデルによって抽出精度に有意差がある**ことが示されている。本notebookは1つのLLMでラベルを作っているため、ここに改善余地がある。

**改善提案**:

1. **ラベル生成LLMを複数使い、ラベル側もアンサンブルする**。上記ベンチマークが示す通りLLM間で抽出精度に差がある。2〜3モデルで独立にラベルを作り、一致度の低い研究には低い重み（sample_weight）を与える、あるいは平均確率を使う。**画像モデルを増やすより、ラベルの質を上げるほうが効く可能性が高い**（学習データが実質4,349件しかないため）。
2. **58件ホールドアウトへの過学習を疑い、統計的に扱う**。gold 58件でのmacro AUCは、稀少所見の陽性が数件しかないため**±0.02程度は普通に動く**。貪欲前向き選択を58件で何十回も回すのは検証セット過学習そのもの。bootstrapで信頼区間を出し、「有意に改善したモデルだけ採用する」ゲートを入れる。
3. **LLMラベル側で大規模CVを組む**。4,349件のソフトラベルを5-foldに切り、そちらでモデル選択・重み決定を行い、58件は最終確認だけに使う。選択と評価のデータを分けるのが原則。
4. **所見間の共起構造を使う**。現在12所見は独立ヘッド。臨床的な共起（半月板+関節液貯留、ACL+骨挫傷など）を、後処理の補正か、マルチタスクの共有表現として入れる。macro AUCでは稀少所見の改善が特に効くので、頻出所見からの情報転移が有効になりうる。
5. **効率トラックを意識してブランチを削る**。各ブランチを1つずつ抜いたときのgold AUC低下を測り（leave-one-branch-out アブレーション）、寄与の小さいブランチを落とす。単一モデルで0.924が出ている以上、30モデル超の構成は正当化しにくい。

### まとめ

**「正解ラベルが58件しかない」という制約に対し、レポートをLLMでソフトラベル化して75倍に増やす**というのが本notebookの思想的な核であり、最も応用の効く発想。一方で、その58件を検証にも使い、そこで貪欲選択まで行っている点は明確な弱点。ラベルを増やす工夫と同じ熱量を、**検証設計を増やす工夫**に向けるべき、というのが今日の学び。

---

## 本日の総括

3本とも「スコアの数字をそのまま信じない」ことがテーマになった。Biohubは**アンサンブルが局所的に悪化しうる**ことを guard で明示的に扱い、S6E8は**スコアの95%が他人の予測**であり、RSNAは**58件のホールドアウトで選択を繰り返している**。共通する教訓は、**改善が本物かノイズかを判定できる検証設計を先に用意すること**。手法の派手さより、そこに投資したnotebookのほうが再現する。
