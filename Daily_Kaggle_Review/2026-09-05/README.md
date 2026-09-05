# 2026-09-05 Kaggle日次レビュー

本日の共通テーマ：**「変更を1点に閉じ込めれば、その効果は測れる」**

3本とも、モデルの新しさではなく **実験設計の作り方** で差がついていました。

- **RSNA** は12所見のうち**内側半月板1つだけ**を差し替え、残り11所見はバイト単位でコピーし、SHA256で「変えていないこと」を証明する。
- **Biohub** は「public 0.940 のベースに対して変えたのは2つ」と宣言し、Configuration Guard で設定ドリフトを実行前に落とす。
- **S6E9** は施策を1つずつ足して OOF AUC の差分を全部表にし、**効かなかった実験（fold数・シード戦略）も正直に書く**。

そして本日は、その裏返しも見つかりました。S6E9のnotebookは**説明文とコードが食い違っています**（「動的シード 42+fold」と書きながら実装は固定42、`early_stopping_rounds` は表で700・実装で500）。結論は変わりませんが、**公開notebookの主張は必ずコードで確認する**という教訓としてそのまま残しておきます。

---

## 1. Biohub - Cell Tracking During Development（固定枠）

| 項目 | 内容 |
|---|---|
| notebook | [🧬 Biohub LB : 941](https://www.kaggle.com/code/analyticaobscura/biohub-lb-941) |
| 原著者 | OZANM.（`analyticaobscura`） |
| スコア | Public LB **0.941**（V1）/ 71 votes / 36分40秒 · GPU T4 x2 |
| ローカル | [`biohub_biohub-lb-941.ipynb`](./biohub_biohub-lb-941.ipynb) |

前日（09-03）扱った Agreement-Gated Dual-Seed Fusion と同じパイプライン系統ですが、プリセットが `harmonic_v3_division_wide` に変わり、**分裂側を広く取りにいく**構成になっています。

### 学べる主要テクニック

1. **8視点の検出TTA** — 上流スクリプトを `read_text() → replace() → write_text()` でモンキーパッチし、反転3種の4視点平均を、90°/270°回転を足して**8視点**に拡張。追加学習ゼロで検出のブレを平均消去する、最もコスパの良い改善手段。
2. **密度適応ギャップ接続** — `GAP_CLOSE_UM=5.0` を固定値ではなく、**周囲3近傍の細胞密度に応じて自動で伸縮**させる（`GAP_DENSITY_GAIN=0.040`, `MAX_STEP_DELTA_UM=0.125`）。混雑した領域では半径を縮めて誤接続を防ぎ、疎な領域では伸ばして見落としを拾う。「1つの閾値を全領域に当てはめない」という発想。
3. **攻めと守りのモデル分離** — safe-div しきい値を 0.14 → **0.25** に上げて分裂候補を広く出し（攻め）、別途学習した DeepCenter が**拒否権（veto）**で偽分裂を却下する（守り）。1つのモデルに両方やらせるより、役割を分けたほうが閾値の意味が明確になる。
4. **Configuration Guard** — 重要パラメータ8個を期待値表と照合し、1つでもズレたら実行前に `RuntimeError`。「36分走って出たスコアが、どの設定のものか分からない」を構造的に排除する30行。
5. **Pipeline Manifest** — 設定値ではなく**実際にロードされたか**を `Path.exists()` と `globals()` で観測して印字。`requested=True / loaded=False` を明示警告する。原著いわく「これまで見つかった静かなバグは全部この印字で見えたはずだった」。
6. **公式指標のオフライン再実装** — `linear_sum_assignment`（ハンガリアン法）で7µmゲート付き二部マッチングを自前実装し、**LBの提出枠を消費せず**に変更を評価する。しかも検証動画は「**分裂がGTに注釈されているもの**」を優先選択する——測りたい現象を含まない検証セットを構造的に排除している。

### 評価指標の要約

**Edge Jaccard（約85%）+ Division Jaccard（約15%）の重み付き和**。Jaccard は `TP/(TP+FP+FN)` なので**過剰検出と見落としを同時に罰する**。ノード照合は7µmゲート付きの最小コスト二部マッチングで、座標は**µm物理空間**（`z=1.625, y=x=0.40625 µm/voxel`）——つまり **XY重心の精度はZの約4倍重い**。分裂は数として極端に少ないが生物学的価値が高いため、Edge項に埋もれないよう独立項として15%を確保している。

### 改善点の考察

**他notebookとの比較**（同コンペのグローバル検索で上位を確認）：

| notebook | Score | このnotebookに無い要素 |
|---|---|---|
| [Biohub Solution](https://www.kaggle.com/code/pilkwang/biohub-solution) | 0.966 | 上位帯の別系統。本notebookとは0.025の差 |
| [Biohub Competition Solution](https://www.kaggle.com/code/boristown/agi-biohub-cell-tracking) | 0.965 | 同上 |
| [Biohub - Track Your Cells Development](https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells-development) | 0.939 | 本notebookとほぼ同帯・別アプローチ |
| [Biohub Harmonic Fusion](https://www.kaggle.com/code/pilkwang/biohub-harmonic-fusion) | 0.940 | 融合方式は同系統 |

**関連文献**：

- **[ARGUS: Accelerated, Robust, General, and Unsupervised Cell Tracking Solutions (arXiv 2607.08297)](https://arxiv.org/abs/2607.08297)** — 適応的検出 + **密Farnebäck オプティカルフロー** + フレーム間線形割当 + **tracklet精緻化による短ギャップ再接続**という構成。本notebookのギャップ接続は「距離と密度」だけで判断しているが、ARGUSは**動きの場（フロー）**を使って「どこに移動したはずか」を予測してから繋いでいる。密集領域での誤接続に効く可能性が高い。
- **[Cell Tracking according to Biological Needs (arXiv 2403.15011)](https://arxiv.org/abs/2403.15011)** — **空間シフトを使ったTTAで位置・運動の密度を推定**し、追跡の偶然的不確実性（aleatoric uncertainty）を明示的にモデル化する。さらに **mitosis-aware な割当問題の定式化**により、長期の矛盾から偽の分裂検出を解消する。本notebookのTTAは「確率を平均する」だけで**不確実性を捨てている**——分散を残せば safe-div の閾値を症例ごとに変えられる。

**改善提案（5点）**：

1. **モンキーパッチの成否を `assert` する。** `_s.replace(_old, _new)` は**置換対象が見つからなくても例外を出さず、元の文字列をそのまま返します**。上流スクリプトが少しでも変われば「8視点TTAのつもりで4視点のまま走る」という静かな劣化が起き、スコアが少し下がるだけで誰も気づきません。`assert _old in _s, "TTA patch target not found"` を各パッチに1行足すだけで防げます。**Configuration Guard と同じ思想を、パッチ適用にも適用すべき**です。
2. **TTAの分散を捨てずに使う（2403.15011の示唆）。** 現在8視点の確率を平均していますが、**視点間のばらつき**は捨てています。「8視点が全員一致した検出」と「4対4で割れた検出」は信頼度が全く違うのに、平均後は区別できません。分散を DeepCenter veto の入力に足せば、閾値0.25を固定値ではなく**確信度に応じた可変値**にできます。
3. **オプティカルフローによるギャップ接続（ARGUSの示唆）。** 現行の密度適応は「周囲がどれくらい混んでいるか」しか見ていません。フローで「この細胞は次フレームでどちらに動くはずか」を推定してから半径を非対称に取れば、同じ半径でも誤接続が減るはずです。
4. **safe-div 0.25 の効果を、Division Jaccard 単体で測る。** 前回0.14→今回0.25で総合0.935→0.941ですが、この+0.006のうち**どれだけが分裂項由来か**が分離できていません。TTAの4→8視点化とギャップ密度適応も同時に入っているので、3つの変更が混ざっています。オフライン validator は既に Division Jaccard を単独で計算できるので、**3つの変更をそれぞれ単独で回せば分解できます**。今日のRSNAがやっている「1点だけ変える」をここにも適用すべきです。
5. **proxy と LB の乖離幅を信頼区間で出す。** 過去の記録では proxy 0.9384 に対し LB 0.934 と約0.004の楽観バイアスがありました。**このバイアスと同程度の改善は、誤差と区別がつきません。** 検証動画を動画単位でブートストラップして Δproxy の信頼区間を出せば、「上がった/下がった」ではなく「有意に上がったか」で判断できるようになります。

---

## 2. Playground Series S6E9 — Predicting Electric Vehicle Purchases

| 項目 | 内容 |
|---|---|
| notebook | [S6E9 Single XGB CV: 0.94488](https://www.kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94488) |
| 原著者 | Дворкин Евгений Владимирович（`evgendvorkin`） |
| スコア | OOF AUC **0.94488** / Public LB **0.94460** / 49 votes（S6E9で最多得票） |
| ローカル | [`playground_s6e9-single-xgb-cv-0-94488.ipynb`](./playground_s6e9-single-xgb-cv-0-94488.ipynb) |

**選定理由**：LB最上位（0.94633）は**OOF予測CSVを共有するだけのnotebook**で手法の中身がなく、0.9462帯もほぼ全てが公開submissionのランク平均でした。対して本notebookは**単一のXGBoost**で0.9446に到達し、しかも各施策の寄与を1つずつ測った表が最初から書かれています。

### 学べる主要テクニック

1. **最初から書かれた ablation study** — ベースライン 0.94204 → digit features **+0.00143** → frequency encoding **+0.00119** → ハイパラ更新 **+0.00022** → 特徴量選択 **±0**。「効いた」を主張する前に測る、が徹底されている。
2. **digit features** — 7つの数値列から `(x // 10**k) % 10` で k=-4〜3 の各桁を抽出し56特徴に。**合成データの生成過程が桁に残す指紋**（丸めパターン、繰り返す小数）を捉える。木モデルは閾値分割しかできないので「小数第2位が7か」のような周期パターンは自力で学べない——**明示的に列にして渡す**。
3. **frequency encoding** — 全列を「その値がtrain+test全体で何割か」に置換。「この値は珍しいか」という**別視点**を1分割で使えるようにする。目的変数を使わないので target encoding より安全。
4. **効かなかった実験の記録** — 10-fold vs 5-fold（±0.00001）、固定シード vs 動的シード（±0.00001）、後処理（Isotonic / Rank / Clip、いずれも改善なし）。**効かなかったことを書き残すのは、効いたことを書くより価値がある。**
5. **スコアを落とさない特徴量削減** — 定数列と完全相関列を除いて 154 → 76 特徴。OOFもLBも1ミリも動かず、モデルだけ半分に。**「同じスコアをより少ない材料で出す」ことに価値を認める**視点。

### 評価指標の要約

**ROC-AUC**。正例スコア＞負例スコアとなるペアの割合。決定的な性質は**順位不変**であること——予測値を単調変換してもAUCは1ビットも変わらないので、**キャリブレーション単体では原理的に1点も上がりません**（原著が「Isotonic / Rank / Clip はすべて改善なし」と報告しているのは、この性質からすれば当然の結果）。効くのは順位を実際に入れ替える操作、すなわち新特徴量・別アーキテクチャ・モデル間ブレンドのみ。digit features も frequency encoding も**非単調変換**だからこそAUCが動きます。**指標の性質から、どこに労力を割くべきかが導かれている**好例。

### ⚠️ 説明文と実装の食い違い（重要）

このnotebookは教材として優秀ですが、**説明文とコードが2箇所ずれています**：

| 項目 | 説明文の主張 | 実際のコード |
|---|---|---|
| シード戦略 | 「動的シード `42 + fold`（43, 44, ..., 52）」 | `current_seed = 42`（**全fold固定**） |
| early stopping | 表では `700` | `early_stopping_rounds: 500` |

原著自身が「固定シード vs 動的シードは差がなかった（±0.00001）」と実験結果を載せているので**結論は変わりません**。ただし**公開notebookのmarkdownを鵜呑みにしてはいけない**という実例として、極めて有益です。数字を引用する前に、必ず該当箇所のコードを読んでください。

### 改善点の考察

**他notebookとの比較**（S6E9 上位帯を確認）：

| notebook | Score | このnotebookに無い要素 |
|---|---|---|
| [S6E9 Electric Vehicle OOF CV 0.94618 LB 0.94633](https://www.kaggle.com/code/najiama/s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633) | 0.94633 | **全モデルのCV設定を `StratifiedKFold(5, shuffle, 42)` に統一**して公開。分割の統一そのものを価値にしている |
| [S6E9 \| Diversity-Boosted Rank Ensemble](https://www.kaggle.com/code/talhatursun/s6e9-diversity-boosted-rank-ensemble) | 0.94628 | 上位ブレンド同士の**ペアワイズ順位相関を実測**し、「大半が >99.7% 相関＝実質同じモデルの繰り返し」を指摘。相関の低い2本を選んで重み付け |
| [Pure LGBM Model CV 0.94587 LB 0.94612](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94587-lb-0-94612) | 0.94612 | 単一LGBM。XGBとは別系統の分割規則＝ブレンド相手として理想的 |
| [S6E9 \| 94.6+ \| Transformer and GBDT Ensemble](https://www.kaggle.com/code/lamhuy8904/s6e9-94-6-transformer-and-gbdt-ensemble) | 0.94608 | 表形式Transformerという**別アーキテクチャ**の多様性 |

**関連文献**：

- **[The Kaggle Grandmasters Playbook (NVIDIA, 2026)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)** — 2026年時点でも「XGBoost / LightGBM / CatBoost が表形式のKaggleをほぼ全勝している」と整理。frequency encoding は**target encoding の補完として機能する**（稀な値が持つ信号を、目的変数を使わずに拾える）と位置づけられている。本notebookは frequency は入れたが target encoding は未導入。
- **[TabArena / TabPFN v2 ベンチマーク（2026）](https://www.codesota.com/tasks/tabular-ml)** — 表形式基盤モデル（tabular foundation model）は**合成データで大量事前学習**され、少数サンプルでのzero-shot性能が伸びている。S6E9のデータ自体が合成である以上、TabPFN系はブレンドの多様性源として検討する価値がある（GBDTとは誤りの出方が根本的に違う）。

**改善提案（5点）**：

1. **説明と実装の乖離をまず直す。** 「動的シード」を本当に有効にすれば（`current_seed = 42 + fold`）、10個のモデルが真に異なる乱数で学習され、**平均化によるバギング効果が強まります**。原著の実験では差がなかったとされていますが、その実験自体が「固定42 vs 固定42」を比較していた可能性があります（コードがそうなっているため）。まず正しく実装してから測り直すべきです。
2. **読み込んだ `orig`（オリジナルデータセット）を実際に使う。** セル4で `EV_Adoption_and_Range_Anxiety_Dataset.csv` を読み込んでいますが、**その後一度も使われていません**。Playground Seriesでは元データを学習に追加するのが定石（データ増加＋合成過程で失われた本物の相関の補完）で、公開解法の多くが +0.0005〜0.001 程度の改善を報告しています。読み込んでいるのに使っていないのは、最も低コストの伸びしろです。
3. **frequency encoding のリーク性を検証する。** `pd.concat([X[col], test[col]])` で train+test 全体から頻度を計算しています。目的変数を使わないので target encoding ほど危険ではありませんが、**CVの中で全体統計を使っているためOOFスコアが楽観的に出る可能性**があります。fold内のtrainだけで頻度を計算した版と OOF AUC を比較すれば、0.94488 のうちどれだけが「本物」かが分かります。CV–LB の乖離（0.94488 vs 0.94460、-0.00028）の一部はここかもしれません。
4. **target encoding を m-schedule で追加する。** frequency（目的変数を使わない）は入っていますが、target encoding（使う）は未導入です。原著自身の「Next Steps」にも挙がっています。**必ずfold内で計算する**という制約を守れば、カテゴリ列の情報を最も直接的に使える手段です。平滑化パラメータ m を複数段（5, 15, 80）で作って全部渡すのが上位解法の定石。
5. **ブレンド相手を「相関の低さ」で選ぶ（Diversity-Boosted の示唆）。** 本notebookの `oof_preds_base.npy` は保存済みなので、あとは混ぜるだけです。ただし**混ぜる相手をスコア順に選ぶのは誤り**で、talhatursun が実測したとおり上位ブレンド同士は99.7%以上順位相関しており、足しても何も増えません。**自分のOOFと最も順位相関が低い**公開OOF（表形式Transformer系が有力）を選ぶべきです。ランク平均ならスケールを気にせず混ぜられます。

---

## 3. RSNA Knee Abnormality Detection（開催中の実コンペ）

| 項目 | 内容 |
|---|---|
| notebook | [RSNA Knee 0.937 \| Weak-Label DINOv2 Meniscus Resid](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid) |
| 原著者 | renta.k（`renta0426`） |
| スコア | Public LB **0.937**（V1）/ 41 votes / 2分51秒 · GPU T4 x2 |
| ローカル | [`competition_rsna-knee-0-937-weak-label-dinov2-meniscus-resid.ipynb`](./competition_rsna-knee-0-937-weak-label-dinov2-meniscus-resid.ipynb) |

**本日の3本の中で、最も設計思想が明確なnotebook**です。0.936の親を一切変えず、12所見のうち**内側半月板1つだけ**を差し替えています。

### 学べる主要テクニック

1. **変更を1所見に閉じ込める** — macro-AUCは12所見の独立な平均なので、1所見の改善は `1/12` の重みでそのまま総合点に乗り、他所見に一切影響しません。**変更を最小単位に閉じ込めることが、そのまま測定可能性になる。** 12所見を同時にいじっていたら、+0.001が何由来か永遠に分かりません。
2. **弱ラベル（weak label）による学習データの拡張** — 4,407 studyのうち**構造化ラベルは58件だけ**。残りは自由記述の放射線科レポートのみ。そこでレポートを言語モデルに読ませ、12個の**確率**に変換した（「断裂が疑われる」→ **1 ではなく 0.8**）。結果、学習可能なstudyが **58 → 4,349件**。**曖昧さを曖昧さのまま目的変数にする**という発想。
3. **臨床知識をアーキテクチャに落とす** — Raptorモデルは `Sagittal / Coronal / Axial × 2` の**6スロット**構成。ACLは矢状断、半月板は冠状断、膝蓋大腿関節は横断が要る——**放射線科医が3方向を見比べる手順**をそのまま入力設計にしている。内側半月板でRaptorの重みを0.60まで上げた根拠も、LBの数字だけでなく解剖学的に説明できる。
4. **所見別の選択的適用** — RadImageNet較正を `_RAD_EXCLUDE = ("Baker's", 'Fracture')` の2所見には**適用しない**。macro平均だからこそ、悪化する所見だけ除外して良いところだけ取れる（micro平均では不可能）。**指標の構造が実装の自由度を決めている**。
5. **ハッシュによる「変えていないこと」の証明** — 親notebook本体・可視テストのUID一覧・親の提出物の SHA256 を事前定義して照合。「変えていないつもり」ではなく**コードで証明**している。
6. **fail fast と safe fallback の使い分け** — 必須のキャッシュ推論は「失敗したら落とす（no fallback）」、任意の上乗せである RadImageNet 較正は「失敗したら生の出力を残す（try/except）」。処理の必須度に応じてエラー戦略を変えている。
7. **mm単位での切り出し** — `CROP_MM = 130.0`。ピクセル数ではなく**物理サイズ**で揃える。撮影装置ごとの `PixelSpacing` の違いを吸収しないと、モデルは病変ではなく**撮影装置を学習してしまう**。医用画像で最も重要な前処理の作法。

### 評価指標の要約

**macro-averaged AUC**（12所見のAUCの単純平均）。これは「**12個の完全に独立した二値分類問題を平等に平均する**」という意味で、3つの帰結を直接生みます：①所見ごとに独立して最適化すべき、②稀な所見も頻繁な所見も等しく 1/12 なので**データ量に比例した労力配分は誤り**、③順位不変なのでキャリブレーション単体では上がらない。micro平均なら有病率の高い変形性関節症で総合点が決まり、見逃すと重大な骨折の性能が埋もれる——macro平均は「**どの所見も臨床的に等しく大事**」という価値判断の数式化です。

**逆算**：総合 0.936 → 0.937 の +0.001 は、1所見の改善が 1/12 に希釈された結果。つまり**内側半月板単体では約 +0.012** の改善があったことになります。

### 改善点の考察

**他notebookとの比較**：

| notebook | Score | このnotebookに無い要素 |
|---|---|---|
| [Head and shoulders, knees and toes](https://www.kaggle.com/code/renta0426/head-and-shoulders-knees-and-toes)（親） | 0.936 | 本notebookの土台。差分は内側半月板のみ |
| [Knee MRI: twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) | 0.924 | **単一モデル・アンサンブルなし・TTAなし**で0.924。本notebookの複雑さの大部分は+0.013のためのもの、という費用対効果の基準点 |
| [RSNA Knee Abnormalities - Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | — | Kaggle公式の**効率性リーダーボード**解説（329 votes）。精度だけでなく**推論コスト**も評価軸になる |
| [RSNA Knee baseline v1](https://www.kaggle.com/code/anhadmahajan06/rsna-knee-baseline-v1) | 0.891 | 512 votes。本コンペの出発点 |

**関連文献**：

- **[PromptRad: Knowledge-Enhanced Multi-Label Prompt-Tuning for Low-Resource Radiology Report Labeling (arXiv 2605.20052)](https://arxiv.org/abs/2605.20052)** — まさに本notebookの弱ラベル生成と同じ問題設定。多ラベル分類を**マスク言語モデリングとして再定式化**し、医学データベースの同義語をカテゴリ表現に組み込むことで、通常のfine-tuningより**遥かに少ない教師データ**で高精度なレポートラベリングを実現。本notebookのLLM読み取りは「プロンプトで確率を出させる」段階に留まっており、同義語知識の注入は行っていない（"suspected" / "cannot exclude" / "questionable" の強度差の扱いが暗黙）。
- **[Weakly Supervised Medical Image Segmentation With Soft Labels and Noise Robust Loss (arXiv 2209.08172)](https://arxiv.org/abs/2209.08172)** — ソフトラベルで学習する際に、通常のBCEではなく **normalized active-passive loss（ノイズ耐性損失）** を使う。膝MRI 17例での骨髄病変の検出・分割で検証済み。ソフトラベルは定義上ノイズを含むので、**損失関数側でもノイズを想定すべき**という指摘。本notebookが損失にどう対処しているかは推論notebookからは読めないが、検討価値が高い。

**改善提案（5点）**：

1. **外側半月板（Lateral Meniscus）に同じ処方を当てる。** 内側半月板に `0.30/0.60/0.10` のランク再構成を当てて約 +0.012 を得たのに、**外側半月板は親のままです**。docstringにも理由が書かれていません。「試したが効かなかった」のか「まだ試していない」のか不明ですが、解剖学的にも撮影断面的にも内側と外側は対称的な所見なので、同じ処方が効く可能性は高い。効けばもう +0.001。**同じ理由で、残り10所見それぞれについても「専用ブレンド重み」を探索できます**——macro平均は所見ごとに分解できるのだから、12回の独立な最適化問題として解くのが最も素直です。
2. **所見別のブレンド重みを、LB手調整ではなくOOFで決める。** `v5=.55, v10=.10, reverse-v5=.15, v8=.20`、`_RAD_ALPHA=0.5`、内側半月板の `0.30/0.60/0.10` ——**これらの重みはすべて手調整であり、その根拠はPublic LBです**。Public LBは全体の一部でしかないので、細かく調整するほど過学習します。**所見別のOOF AUCを最大化する重みを、fold内で求める**べきです（12所見 × 4アームなら48個のパラメータですが、非負・和1の制約下の凸最適化なので `scipy.optimize` で解けます）。
3. **弱ラベルの「疑わしさ」の粒度を上げる（PromptRadの示唆）。** 現在は「suspected → 0.8」という単一の写像です。しかし放射線科レポートには "cannot be excluded"（0.3?）、"highly suspicious for"（0.9?）、"questionable"（0.5?）、"consistent with"（0.95?）といった**確信度の階層**があります。同義語辞書を使ってこの階層を明示的に符号化すれば、ラベルの質が上がります。ラベルの質はモデルの上限を決めるので、アンサンブルを1本増やすより効く可能性があります。
4. **ソフトラベル前提のノイズ耐性損失を使う（arXiv 2209.08172の示唆）。** 目標値が 0.8 のとき、通常のBCEは「0.8にぴったり合わせよ」という強い要求を出します。しかし 0.8 自体がLLMの推定でありノイズを含むので、**そこまで厳密に合わせる意味がありません**。normalized active-passive loss のようなノイズ耐性損失なら、外れたラベルに引きずられにくくなります。学習側の変更なので推論notebookでは検証できませんが、次に学習を回すときの第一候補。
5. **`np.zeros` のモンキーパッチを、正攻法に置き換える。** セル1の capture shim は技巧としては見事ですが、上流が `np.zeros` を `np.empty` に変えただけで**静かに壊れます**（例外は出ず、キャッシュが取れないだけ）。「親を絶対に変えない」という制約がこの複雑さを生んでいますが、その制約自体は**ハッシュ照合で担保されているのだから**、親を1行だけ変えてキャッシュを返すようにし、その版のハッシュを新たに固定するほうが安全です。最低でも、捕捉に失敗したときに例外を投げる `assert` は必要です（現在は候補リストが空のまま後続に進む可能性があります）。

---

## 本日のまとめ

3本すべてが「**変更を1点に閉じ込めて、その効果を測る**」という同じ結論に、別々のコンペで独立に到達していました。

- RSNA：12所見のうち1所見だけを差し替え、残りをハッシュで固定 → **+0.001 の出所が完全に特定できる**
- Biohub：Configuration Guard と Pipeline Manifest で「設定した値」と「実際に効いた状態」を分離 → **静かなバグが見える**
- S6E9：施策ごとの OOF AUC 差分を全部表にする → **どの施策がいくら稼いだかが分かる**

一方で、3本に共通する弱点も同じでした。**重み・閾値のほとんどが Public LB を見た手調整**です（Biohubの `safe_div=0.25`、RSNAの `0.30/0.60/0.10`、S6E9のハイパラ）。実験の管理はここまで厳格なのに、**最後のパラメータ選択だけが LB 依存**という非対称が残っています。ここをOOFベースの最適化に置き換えるのが、3本すべてに共通する次の一手です。

そして本日の裏テーマとして、**S6E9のnotebookは説明文とコードが食い違っていました**（動的シードと書いて固定シード、700と書いて500）。上位notebookであっても、主張は主張であって検証済みの事実ではありません。**数字を引用する前に、必ず該当箇所のコードを読む。** これは今日最も実用的な学びかもしれません。
