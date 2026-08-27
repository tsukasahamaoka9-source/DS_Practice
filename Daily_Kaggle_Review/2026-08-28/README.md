# 2026-08-28 Kaggle日次レビュー

今日の3本は、偶然にも**「学習をせずにスコアを取る」3つの異なる形**が並びました。Biohubは既存検出器の使い方を極限まで詰め、Playgroundは他人の提出を混ぜるだけ、RSNAは学習済み4系統を段階的に融合する。**「モデルを強くする」以外にスコアを動かす方法がこれだけある**、という日です。

---

## 1. 🧬 Biohub - Cell Tracking During Development / `biohub-sdw60`

| | |
|---|---|
| コンペ | [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) |
| notebook | [biohub-sdw60](https://www.kaggle.com/code/arnav170/biohub-sdw60) |
| 原著者 | ARNAV170 |
| Public Score | **0.927**（公開notebookの最上位帯） |
| ローカル | [`biohub_biohub-sdw60.ipynb`](./biohub_biohub-sdw60.ipynb) |

### 学べる主要テクニック

- **Harmonic bidirectional fusion** — エッジのスコアを t→t+1 と t+1→t の両方向で計算し、**調和平均**で融合する。算術平均と違い「両方が高いときだけ高い」という論理AND的な性質を持ち、片方向だけの誤対応を潰せる。
- **frame-retention guard** — アンサンブルが主モデルより検出候補を減らしたフレームだけ、主モデル単独に巻き戻す。**全体一括ではなくフレーム単位で安全側に倒す**部分適用の発想。
- **8-view TTA + Dual-seed アンサンブル** — 学習をやり直さずに予測分散を下げる王道。ただし推論時間はほぼ2倍。
- **Configuration Guard / Pipeline Manifest** — 設定のドリフトを実行時に検知し、最後に「設定」ではなく**実際に何が有効だったか**を印字する。silent failure対策として最も安価で効果的。
- **公式指標のローカル再実装** — hold-outした学習動画で公式指標を自前計算。リーク検査（検証動画がテストに混ざっていないか）まで明示的に行っている。

### 評価指標の要約

ノードを7 µm以内でハンガリアン法により1対1対応させたうえで計算する、**エッジのJaccard係数**（`TP/(TP+FP+FN)`）に、過剰ノード数ペナルティ（係数0.1）と分裂イベント重み（0.1）を加えた調整版。**ノードを1つ落とすと入るエッジと出るエッジが同時に壊れるため、検出の見逃しは2倍以上のコストになる** — このnotebookの設計（消失に重いペナルティ1.5、gap closing、retention guard）はすべてこの一点から導かれている。

### 改善点の考察

**他notebookとの比較**（同コンペのBest Score上位を確認）:

- [`Metric_hack_last_call_update`](https://www.kaggle.com/code/kaiwalyaatulraut/) / `Biohub Solution` は **0.965〜0.966** と一段上だが、タイトルどおり指標の穴を突く性質のもの。sdw60はこの系統を採用していない。**学習目的では正しい選択**だが、「指標のどこに穴があるか」を理解しておくことは、逆に**自分の指標設計を守る**うえで有益。
- [`🧬Clean Approach + Lightweight Local CV | No Hack`](https://www.kaggle.com/code/) (0.908, 196票) は**軽量なローカルCV**を売りにしている。sdw60の検証は胚タイプごとに2本のhold-outのみで、**検証セットが小さく分散が大きい**可能性がある。
- `Biohub Cell Tracking V4 UNet ILP Reproduction` (0.896) 系は検出器アーキテクチャ自体に手を入れている。sdw60は**検出器は固定して後処理と融合だけで戦っている**ため、検出段の伸びしろが手つかずで残っている。

**関連文献**: [Higher-Order Cell Tracking Transformer](https://arxiv.org/html/2607.11754)（2026年7月）は、隣接フレームのペア単位ではなく**3フレーム以上の高次関係**をTransformerで直接扱い、CTC系ベンチマークで首位を報告している。本notebookのbidirectional fusionは「2フレーム間の対称性」を使う手法であり、その自然な一般化が高次アソシエーションにあたる。また [ELEPHANT](https://elifesciences.org/articles/69380) はインクリメンタル学習で少量アノテーションから3D系譜を構築する枠組みで、ラベル効率の観点で参考になる。

**改善提案**:

1. **双方向の融合を3フレーム以上へ拡張する。** 現在は t↔t+1 の対称性のみ。t-1, t, t+1 の三つ組で整合性を要求すれば、分裂直後の誤対応（分裂で見た目が急変する瞬間）を大きく減らせる可能性がある。上記の高次Transformer論文が示す方向。
2. **検証セットを増やして分散を測る。** 胚タイプごと2本は少なく、パラメータ変更の効果が検証ノイズに埋もれる。実際 `DIVISION_WEIGHT` を4通り変えても実LBが動かなかったという記録は、**検証が効果を検出できていない**可能性も示唆する。まず「同じ設定を2回走らせたときのスコア幅」を測るべき。
3. **分裂イベント専用の指標を分けて監視する。** 分裂は数が少ないため、全体Jaccardの動きに埋もれる。分裂だけのprecision/recallを別に出せば、`DIVISION_WEIGHT` の調整が初めて意味を持つようになる。
4. **検出しきい値を所与とせず掃引する。** `0.96875` は固定値。gap closingで回収する設計なら、**しきい値をもっと下げてFNを減らし、DeepCenter vetoでFPを削る**というバランスの再配分が試せるはず。
5. **8-view TTAのコスト効果を測る。** 推論時間がほぼ2倍になる。4-viewに戻した分の時間で3つ目のシードを足すほうが、同じ計算予算で利得が大きい可能性がある。

---

## 2. 📱 Playground S6E8 - Predicting Smartphone Addiction / `S6E8: 🚀🔥0.97123`

| | |
|---|---|
| コンペ | [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)（8/31終了・残り4日） |
| notebook | [S6E8: 🚀🔥0.97123](https://www.kaggle.com/code/itzzomkar/s6e8-0-97123) |
| 原著者 | OMKAR KADAM |
| Public Score | **0.97123**（実行時間23秒・**モデル学習ゼロ**） |
| ローカル | [`playground_s6e8-0-97123.ipynb`](./playground_s6e8-0-97123.ipynb) |

### 学べる主要テクニック

- **Rank-Gauss変換** — 予測を「順位 → 一様分布 → 正規分布（`norm.ppf`）」と写してから平均する。単純な順位平均より優れるのは、**正規分布に写すと端（確信度の高い領域）の間隔が引き伸ばされ、上位での意見の食い違いが正しく重く扱われる**ため。
- **AUCが順序しか見ない性質の利用** — 予測値を単調変換してもAUCは1ビットも変わらない。だからキャリブレーションが不要で、スケールの違うモデルを対等に混ぜられる。
- **`glob(..., recursive=True)` による提出の動的収集** — パスを直書きせず、Inputを差し替えるだけでブレンド対象が変わる。`sample_submission` の除外も忘れていない。
- **（反面教師として）根拠のない重み付け** — 重み0.35/0.15の割り当てが**ファイル名の文字列マッチ**で決まっている。

### 評価指標の要約

`addicted_label` の二値分類、**ROC-AUC**。AUCは「無作為に選んだ正例と負例で、正例のスコアが高い確率」に等しく、**順序のみに依存する**。この一点がRank-Gaussブレンドを成立させている全根拠。

### 改善点の考察

**他notebookとの比較**（同コンペのBest Score上位を確認）:

- [`[S6E8] Top 20 Formula: Dual Master Rank Blend`](https://www.kaggle.com/code/) (0.97128) と `S6E8: Elite Rank Average Ensemble` (0.97126) が僅かに上。いずれも同系統のランクブレンドで、**上位陣が全員ほぼ同じ手法に収束している**。差は0.00005程度で、これは実質的にノイズの範囲。
- 対照的に [`Why Every S6E8 Notebook Above 0.97110 Overfits`](https://www.kaggle.com/code/) (0.97115, 64票) と `S6E8: will your 0.971 survive the private split?` (50票) は、**この収束そのものが危険信号だと明示的に警告している**。上位帯の差がPublicのノイズに由来するなら、Private LBでの順位は事実上くじ引きになる。
- 学習を伴う系統（`S6E8: CatBoost` 0.96813、`S6E8: LGBM` 0.9676）は0.003ほど下。**ブレンドが稼いでいる0.003が本物の汎化なのか、Publicへの適合なのかは、このnotebook単体からは判別できない。**

**関連文献・記事**: Kaggleの上位解法writeup（[4th Place: Why "Less is More" (Stumps & Rank Ensembling)](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/4th-place-solution)）では、生確率の平均は**わずかなキャリブレーション差が平均を強く歪める**ためランク平均を採用した、と同じ論理が述べられている。加えて同writeupは **OOFとPublic LBの乖離幅を監視し、閾値を超えた実験は破棄する**という運用を紹介しており、本notebookに最も欠けている部分がまさにそこ。

**改善提案**:

1. **重みをファイル名ではなくOOFで決める。** これが最大かつ最も明白な改善点。OOF予測があれば `scipy.optimize` や hill-climbing でAUCを直接最大化できる。名前に "elite" が入っているかは性能と無関係。
2. **OOFが無いなら、せめて予測間のSpearman相関で重みを決める。** 相関の低い（＝多様な）予測に厚く配分するほうが、名前マッチより遥かに理屈が通る。相関0.99のモデルを5本混ぜても実質1本と変わらない。
3. **CV-LB乖離をモニタする運用を入れる。** 上記writeupの「乖離が一定幅を超えたら捨てる」ルールは、そのままこのコンペに輸入できる。
4. **`merge(how='inner')` の行数を検証する。** どれか1つの提出でidが欠けると行が落ち、提出が不正になる。`assert len(merged) == len(sample_sub)` の一行を足すだけで防げる。
5. **このスコアを自分の実力の指標にしない。** 学習ゼロで0.97123に届くという事実は、**Public LB上位帯がほぼ情報を持っていない**ことの証拠でもある。学習を伴う自作モデルのOOF AUCこそが、追うべき数字。

---

## 3. 🦵 RSNA Knee Abnormality Detection / `rsna-base`

| | |
|---|---|
| コンペ | [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)（10/22終了・残り2か月、賞金$77,000） |
| notebook | [rsna-base](https://www.kaggle.com/code/anvithpothula/rsna-base) |
| 原著者 | ANVITH POTHULA（DINOsaur V10 / Roman Tamrazov 版の忠実な移植と明記） |
| Public Score | **0.936**（公開最上位帯） |
| ローカル | [`competition_rsna-base.ipynb`](./competition_rsna-base.ipynb) |

### 学べる主要テクニック

- **レポートのLLMソフトラベル化** — 構造化ラベルは4,407件中**58件だけ**。読影レポートを言語モデルに読ませ、12所見それぞれを0/1ではなく**確率**（「疑われる」→0.8）に変換して4,349件の訓練データを作っている。**このコンペの本丸は画像モデルではなくラベル生成**。
- **物理単位（mm）クロップ** — `CROP_MM = 130.0`。多施設・多機種のデータでピクセル基準に切ると映る解剖範囲が装置ごとに変わる。DICOMの `PixelSpacing` を読んで常に同じ物理範囲を切り出す。
- **3断面 × 2方向の6スロット構成** — 矢状断はACL・半月板、冠状断はMCL・内外側OA、軸位断は膝蓋大腿・関節液。**所見ごとに見える断面が違う**という臨床知識がそのまま設計になっている。
- **所見単位で融合重みを変える** — RadImageNet系統をベーカー嚢腫と骨折の2所見だけブレンドから除外（`_RAD_EXCLUDE`）。macro平均AUCでは各所見が独立に効くので、この単位での最適化が素直に効く。
- **段階的submission更新 + フェイルセーフ** — 4系統を順に走らせ、各段で `submission.csv` を更新。最終融合は `try/except/finally` で囲み、失敗したら**バックアップから直前の提出を復元**する。提出枠が有限なCode Competitionの必須作法。
- **重みのSHA-256検証** — 外部データセットは所有者が差し替えうる。「昨日と同じモデルを使っている」は確認しない限り保証されない。

### 評価指標の要約

12所見それぞれのROC-AUCを**単純平均**（macro-averaged AUC ROC）。**症例数の少ない骨折やベーカー嚢腫が、頻出するACLや変形性関節症とまったく同じ重みを持つ**。頻出所見だけを磨いても上限は 11/12 ≒ 0.917 相当で、**稀な所見のAUCを1点上げるほうがリターンが大きい** — このコンペの戦略の骨格を決めている性質。

### 改善点の考察

**他notebookとの比較**（同コンペのBest Score上位を確認）:

- [`Head and shoulders, knees and toes`](https://www.kaggle.com/code/) (0.936) と [`RSNA Knee | DINOsaur V4 🦖`](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4) (0.936) は同スコアで、いずれも**同じDINOsaur系統の派生**。公開上位帯が単一系譜に強く収束している。
- [`Knee MRI: twelve findings from a single model`](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) (0.924, 85票) は**アンサンブルもTTAも無しの単一モデル**で0.924。本notebookの0.936との差はわずか0.012であり、**4系統の融合で稼いでいるのはたった1.2ポイント**。効率トラック（Efficiency Prize）を狙うなら単一モデルのほうが圧倒的に有利。
- `RSNA Knee Abnormalities - Efficiency LB` (269票) が示すとおり、このコンペには**AUCと実行時間の両方で評価される効率賞（$18,000）**があり、そちらは公開上位が薄い。**本notebookの2m55sという実行時間は実は効率賞の射程内**で、狙い目として見過ごされている可能性がある。

**関連文献**: [Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities](https://www.nature.com/articles/s41467-024-51888-4)（Nature Communications）は、まさに**同じ12所見**を対象に、複数MRIシーケンス間で**co-plane attention**を学習する手法を提案している。本notebookは6スロットを独立に処理して後段で混ぜているのに対し、この論文は**断面間の対応関係をモデル内部で明示的に学習**する。また [9所見・14,962件・5施設の多施設検証研究](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(25)00467-5/fulltext)（eClinicalMedicine）は、施設間の汎化性能低下を定量化しており、19か国からデータが集まる本コンペのPrivate LB挙動を考えるうえで示唆的。

**改善提案**:

1. **断面間の融合をモデル内部に移す。** 現在は6スロットを独立推論して後段で混ぜている。上記のco-plane attentionのように、**断面をまたいだattentionを学習段階で持たせる**ほうが、半月板のように複数断面の情報を統合して初めて確定する所見に効くはず。
2. **稀な所見（骨折・ベーカー嚢腫）に的を絞る。** macro平均である以上、ここが最大のレバレッジ。ベーカー嚢腫は膝窩にあり130 mmクロップから外れやすい — **この2所見だけクロップ範囲や断面の重みを変えた専用ヘッド**を用意する価値がある。RadImageNet系統をこの2つで除外している事実は、裏を返せば**まだ誰も解けていない**ことの証拠。
3. **効率トラックを狙う。** 単一モデル0.924と4系統0.936の差は1.2ポイント。効率スコアは実行時間で割られるため、**セル1だけ、あるいはセル2だけで提出する軽量版**が本線より高い期待値を持つ可能性がある。公開notebookが薄い領域でもある。
4. **多施設汎化を疑う。** 19か国以上・多言語レポートから構成されるデータで、Public/Privateの施設分布が同じ保証はない。**施設（またはStudyの由来）でグループ化したCV**を組まないと、CVもPublic LBも楽観的になりうる。
5. **ラベル生成そのものを検証する。** ラベルがLLMによるレポート解釈である以上、**モデルが病変ではなくレポートの書き癖を学習している**リスクがある。構造化ラベルのある58件は貴重な「真の正解」なので、そこでのAUCを別途監視すべき。
6. **単調でない後処理を探す。** AUCは順序しか見ないため、**単調なキャリブレーションはスコアを1ミリも変えない**。88特徴量transformerが効いているとすれば、それは所見間の順序を組み替えているから。ここを意識して設計すれば伸びしろがある。

---

## 📌 今日のまとめ

3本を貫くのは **「指標の性質を知ることが、そのまま手法選択になる」** という一点でした。

- AUCが**順序しか見ない** → だからRank-Gaussブレンドが成立し、単調なキャリブレーションは無意味になる（S6E8・RSNA）。
- macro平均が**各所見を等価に扱う** → だから稀な所見に投資し、所見単位で融合重みを変えるのが正解になる（RSNA）。
- Jaccardが**エッジ基準** → だからノードの見逃しが2倍のコストになり、消失に重いペナルティを置く設計が導かれる（Biohub）。

そしてもう一つ。**Biohubのconfiguration guard / pipeline manifest / label-free監査、RSNAのSHA-256検証 / フェイルセーフ** — 上位notebookのコードの半分近くが「正しく動いていることを自分で証明する仕組み」に費やされていました。**モデルは静かに壊れる**（エラーを出さずに少し悪い結果を返す）という前提に立つと、これらは贅沢品ではなく必需品です。今日の3本で最も持ち帰る価値があるのは、おそらくここです。

逆に S6E8 は、**Public LBスコアが実力の指標として信用できなくなる瞬間**を見せてくれました。学習ゼロで最上位帯に並べるということは、その帯にもう情報が残っていないということ。追うべきは自作モデルのOOFです。
