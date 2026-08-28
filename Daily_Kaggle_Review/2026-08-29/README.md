# 2026-08-29 Kaggle日次レビュー

今日の3本は、偶然にも **「アンサンブルの重みを、どこまで信じて最適化してよいか」** という共通のテーマで並んだ。
Biohubは「機械的な自己検査で設定ドリフトを封じる」方向、S6E8は「強い正則化で重みの自由度を意図的に殺す」方向、
RSNAは「重みを攻めに行き、そのリスクを明示して選択可能にする」方向。同じ問題への3つの異なる回答として読むと面白い。

---

## 1. 【固定枠】Biohub - Cell Tracking During Development — 「Biohub 0.928 LB」

- **コンペ**: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)（Research / Code Competition、$60,000、約2,836チーム、残り1か月）
- **notebook**: [Biohub 0.928 LB](https://www.kaggle.com/code/evgendvorkin/biohub-0-928-lb) / 原著者: **ДВОРКИН ЕВГЕНИЙ ВЛАДИМИРОВИЧ (evgendvorkin)**
- **Public Score**: **0.928**（V15、62 votes、Silver）— 現時点で公開notebookの実質トップ帯
- **ファイル**: [`biohub_biohub-0-928-lb.ipynb`](./biohub_biohub-0-928-lb.ipynb)

### 学べる主要テクニック

1. **Configuration Guard** — 重要な設定値8個を期待値辞書とハードコード照合し、1つでもズレたら実行開始10秒で `RuntimeError`。30分のGPU推論を無駄にせず、フォーク時の「なぜかスコアが再現しない」を構造的に潰す。
2. **双方向エッジ融合の調和平均** — `p_fwd` と `p_rev` を `2ab/(a+b)` で融合。小さい方に強く引っ張られる性質を使い、「片思いリンク」を積極的に落としてFPを削る。
3. **frame retention guard** — アンサンブル後の候補数が主モデル単体を下回ったフレームだけ、主モデルの結果へロールバック。平均化の副作用（両モデル同時の閾値割れ）をピンポイントで打ち消す。
4. **DeepCenter veto** — gap closingで復元した合成ノードを、独立に学習した中心検出モデルに再確認させ、否定されたら取り消す。「繋げれば得」という誘惑をFPノードの増加から守る二重チェック。
5. **Pipeline Manifest** — 最終セルで「configではON」と「実体としてロード済み」を対比出力。secondaryモデルのパスが1文字違うだけで single-seed に静かに降格する、というサイレント・デグレードを可視化する。
6. **µm換算での距離判定** — z解像度がxyの4倍粗い（1.625 vs 0.40625 µm/voxel）ため、ボクセル距離のままだとz方向を4分の1に過小評価する。全ての閾値を物理単位で扱う。

### 評価指標の要約

**Edge Jaccard（約85%）+ Division Jaccard（約15%）の加重和**。ノードのマッチングはフレームごとのハンガリアン法（7µmゲート）で行い、
エッジは「両端がGTノードにマッチし、かつGT側に同エッジが存在」ならTP。アノテーションが疎で細胞総数が不明なため、
「たくさん出せば得」を防ぐ **node-count penalty**（予測ノード数が `estimated_number_of_nodes` を超えると減点）が組み込まれている。
分裂は全動画の約44%にしか存在しない稀イベントなので、Edge Jaccardに埋もれないよう別枠15%で保護されている。
このnotebookは、高い検出閾値（0.96875）＋短トラック除去でFPノードを抑えつつ、`DISAPPEARANCE_WEIGHT=1.5 / APPEARANCE_WEIGHT=0.0` という非対称なILP重みでエッジFNを潰す、という**指標の重み配分に忠実な設計**になっている。

### 改善点の考察

**他notebookとの比較**（同コンペの上位帯を確認: [Biohub Harmonic Fusion 0.928](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion)、[Biohub Cell Tracking 0.926](https://www.kaggle.com/code/kunaldesale2408/biohub-cell-tracking)、[0.926-biohub-divsub](https://www.kaggle.com/code/rockerritesh/0-926-biohub-divsub)）:
上位帯が0.926〜0.928に**団子状態で密集**しており、しかも構成要素（TemporalUNet3D + Node Transformer + ILP）はほぼ共通。
差はすべて後処理の閾値と融合方法にある。裏を返せば、**この構成のままでの伸びしろはほぼ枯れている**ということでもある。

**関連文献**: [Higher-Order Cell Tracking Transformer (HOCT, arXiv:2607.11754)](https://arxiv.org/abs/2607.11754) は、
リンク候補（エッジ）同士を3D幾何事前分布のもとで相互にattentionさせる **edge-centric** なアーキテクチャで、
深層の事前学習済み画像エンコーダ無しに Cell Tracking Challenge と細菌分裂ベンチマークでSOTAを達成している。
また [Trackastra (arXiv:2405.15700)](https://arxiv.org/pdf/2405.15700) は、Transformerで直接リンクを予測し ILP を軽量化する方向。

**改善提案**

1. **エッジ間の相互作用をモデル化する**（HOCT的アプローチ）。現状のNode Transformerは各エッジを独立にスコアリングし、相互作用はILPの制約でしか表現されていない。「同じ親から出る2本のエッジ」を同時に見るattentionを入れれば、分裂の検出（15%枠）が構造的に改善しうる。ILP前の確率がすでに構造を知っている状態を作れる。
2. **node-count penaltyを損失に直接組み込む**。現在は検出閾値 0.96875 という**1つのスカラー**でFP/FNのバランスを取っているが、`estimated_number_of_nodes` は動画ごとに与えられている。**動画ごとに閾値を動的に決める**（予測ノード数が推定値に一致するよう二分探索する）だけで、全動画一律の閾値より確実に良い。実装コストが低く、最も費用対効果が高い提案。
3. **調和平均以外の融合関数を試す**。`2ab/(a+b)` は「片方が低ければ落とす」に特化しているが、幾何平均 `√(ab)` や一般化平均 `((a^p+b^p)/2)^(1/p)` の `p` をローカルバリデータで探索する余地がある。`p` を1つ増やすだけで探索空間が広がる割に、実装は数行。
4. **secondary seedを2→3以上に増やす**。現在は2seedだが、`retention guard` の枠組みは3seed以上にそのまま拡張できる。GPU時間との相談だが、団子状態を抜けるには「同じ工夫の精緻化」より「単純な多様性の追加」が効きやすい局面。
5. **分裂の時間的一貫性を使う**。細胞周期には典型的な長さがある。「直前に分裂したばかりの細胞が数フレーム後に再び分裂する」候補には、時間間隔に基づくペナルティを掛けられる。`SAFE_DIV_*` は現状すべて**空間的**な閾値なので、**時間軸の事前知識が丸ごと未使用**。

---

## 2. 【Playground】Playground Series S6E8 — 「S6e8 Public Ensemble」

- **コンペ**: [Predicting Smartphone Addiction (S6E8)](https://www.kaggle.com/competitions/playground-series-s6e8)（3,196チーム、**残り3日**）
- **notebook**: [S6e8 Public Ensemble](https://www.kaggle.com/code/kirill0212/s6e8-public-ensemble) / 原著者: **CSTDY (kirill0212)**
- **Public Score**: **0.97120**（V4）
- **ファイル**: [`playground_s6e8-public-ensemble.ipynb`](./playground_s6e8-public-ensemble.ipynb)

### 学べる主要テクニック

1. **自分でモデルを1つも学習しないスタッキング** — 公開OOFライブラリ（szymonkapiski 47モデル、adarsh1077、dariushafshar golem library ほか）と他人のnotebook出力を数十本読み込むだけで、200行未満で LB 0.97120。
2. **極端に強い正則化のメタモデル** — `LogisticRegression(C=0.00599484)`。`C` は正則化強度の逆数なので、これは「係数をほぼゼロに縮める」設定。高相関なOOFを大量に入れたときの**多重共線性による係数の暴走**を、自由度を殺すことで封じている。
3. **形式に寛容なローダ** — CSV / Parquet / `.npy` を吸収し、「float型かつユニーク値2超の最後の列」をスコア列と推定。`|値| <= 1.1` なら確率、超えていればロジットと自動判定。読めなかったファイルは `SKIP` して先へ進む。
4. **ロジット空間での融合** — `prob_to_logit` で `±30` クリップ付きロジット化。AUCは単調変換不変なのでスコア上は無害だが、確率のまま平均すると 0.999 と 0.9999 の差（ロジットで 6.9 vs 9.2）が潰れる。
5. **foldごとの係数を毎回印字** — fold間で係数がブレる＝メタモデルが不安定というサイン。スコアだけでなく**重みの安定性を監視**する。

### 評価指標の要約

**ROC AUC**（二値分類）。予測の順位だけを見るため、ロジット化・順位化といった単調変換ではスコアが一切変わらず、ブレンド設計の自由度が非常に高い。
逆に言えば「確率のキャリブレーションを直しても1ミリも上がらない」ので、努力を投じる先を間違えやすい指標でもある。

### 改善点の考察

**他notebookとの比較**: 上位帯（[regime-calibrated rank fusion 0.97127](https://www.kaggle.com/code/atakanaldemir/s6e8-regime-calibrated-rank-fusion-lb-0-97127)、
[top-20 formula dual master rank blend 0.97128](https://www.kaggle.com/code/souvikdbiswas/s6e8-top-20-formula-dual-master-rank-blend)、
[elite rank average 0.97126](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97126)）と比べると、
本notebookの 0.97120 との差は **0.00008 程度**。上位が採用していて本notebookが採用していないのは主に、
(a) **レジーム依存の重み**（欠損パターンや特徴量帯ごとに別の重みを使う）、(b) **順位空間とロジット空間の両方をメタモデルに与える二重表現**、(c) **多シード平均**（本notebookは `SEEDS=[42]` の1本のみ）。
一方、注目すべき対抗notebookが [S6E8: will your 0.971 survive the private split?](https://www.kaggle.com/code/georgymamarin/s6e8-will-your-0-971-survive-the-private-split)（50 votes、全37版）で、
「公開LB上位帯の 0.9712 前後の差は private split では消える可能性が高い」ことを、過去7ボードの public↔private 対応から検証している。**残り3日で本当に効くのは、0.00008の追い上げではなく、この分析の方**。

**関連文献**: [How Ensemble Learning Balances Accuracy and Overfitting: A Bias–Variance Perspective on Tabular Data (arXiv:2512.05469)](https://arxiv.org/html/2512.05469v1) は、
スタッキングがベース学習器の**多様性とデータ量**に依存し、条件が揃わないと最良単体モデルを超えないことをバイアス–バリアンス分解で示している。
NVIDIAの [Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) も、
「相関の高いOOFを大量に入れた最終ロジスティック回帰は非常に容易に過学習する」と明記している。

**改善提案**

1. **`SEEDS` を複数にする**（実装コスト1行）。`SEEDS = [42, 2024, 777]` にするだけ。CV分割の引きの良し悪しを平均で薄められ、private側の分散が確実に下がる。コードは既にリスト対応済み。
2. **OOFプールの相関を可視化してから枝刈りする**。数十本のうち、相関 0.999 超のペアは実質「同じ意見の重複投票」。プロヴェナンス（誰の予測を誰がブレンドしたか）を辿り、**他人のブレンド結果がプールに混入していないか**を確認すべき。混入していると、そのメンバーは「データのモデル」ではなく「プールのモデル」になり、新情報をほぼ持たない。
3. **順位表現とロジット表現の両方をメタモデルに与える**。ロジットは自信の強さ（テール情報）を保持し、順位はスケール差に頑健。両方を列として渡せば、メタモデルが状況に応じて使い分けられる。上位notebookが採っている定石。
4. **最終2提出の片方を保守側に振る**。残り3日という段階では、public最高の 0.97128 を追うより、**public 0.9710 前後でもfold間の係数が安定しているブレンド**を1本確保する方が期待順位は上がりやすい。上記の "will your 0.971 survive" notebookの分析はそのための材料そのもの。
5. **`|値| <= 1.1` ヒューリスティックの安全化**。予測が全て0付近に集中しているロジット出力のモデルは、誤って「確率」と判定されて二重変換される。読み込み時に各ファイルのmin/max/平均を印字するだけで、事故を目視で捕まえられる。

---

## 3. 【実コンペ】RSNA Knee Abnormality Detection — 「RSNA Knee | Crazy LB Tune」

- **コンペ**: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)（Research / Code Competition、$77,000、約2,606チーム、残り2か月）
- **notebook**: [RSNA Knee | Crazy LB Tune](https://www.kaggle.com/code/tamerlanomralinov/rsna-knee-crazy-lb-tune) / 原著者: **TAMERLAN OMRALINOV**（上流: [Roman Tamrazov — DINOsaur V10](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4)）
- **Public Score**: **0.936**（V3、19 votes）
- **ファイル**: [`competition_rsna-knee-crazy-lb-tune.ipynb`](./competition_rsna-knee-crazy-lb-tune.ipynb)

### 学べる主要テクニック

1. **LLMによる読影レポート→ソフトラベル変換** — 4,407 studyのうち構造化ラベルは**58件のみ**。残りはフリーテキストのレポートしかない。言語モデルにレポートを読ませ、12所見それぞれを**確率**（「断裂が疑われる」→ 1 ではなく 0.8）に変換して4,349件の学習データを作った。58件は学習に一切使わず評価専用に温存（macro-AUC 0.9167）。**このコンペの本質はモデル選びではなくラベル生成**。
2. **rank空間での3系統融合** — DINOv2/v3、RadImageNet(ResNet-50)、CoAtNet 2本を、それぞれパーセンタイル順位に変換してから加重和。macro-AUCは順位しか見ないので、確率スケールを揃える手間なしに安全に混ざる。
3. **所見ごとに異なる補完重み** — `MCL 0.16 / Synovitis 0.16 / Medial OA 0.10 / PF OA 0.12 ...`、さらに `Baker's` と `Fracture` はRadImageNet系の寄与を**ゼロにして除外**。macro指標は所見ごとに独立なので、所見単位の最適化が干渉しない。
4. **臨床共起事前知識による補正** — `CLINICAL_NEIGHBORS`（ACL断裂 ↔ 骨挫傷・外側半月板、滑膜炎 ↔ 関節液貯留・Baker嚢腫 …）を辞書化し、共起しやすい所見同士でスコアを引き上げ／引き下げる。ドメイン知識の後付け注入。
5. **物理単位（130mm）クロップ** — MRIは装置・患者ごとに `PixelSpacing` が異なる。ピクセル数固定で切ると装置が変わるだけで視野が変わる。医療画像では必須の前処理。
6. **リスクを分類した7プロファイル + 1行スイッチ** — `direction_005`/`bootstrap_shrunk`（低分散）→ `raptor065`/`clinical_moderate`/`hybrid_public`（public狙い）→ `aggressive`/`aggressive_residual`（最高リスク）。全部を `/kaggle/working` に書き出し、最終セルの1行で切り替える。元のV10出力も `submission_v10_exact.csv` として対照保存。

### 評価指標の要約

**12所見それぞれの ROC AUC を平均した macro-AUC**。所見ごとの陽性率が極端に違う（変形性関節症は多く、骨折は稀）ため、
サンプル単位で平均すると頻度の高い所見だけで数字が決まってしまう。所見ごとにAUCを出してから平均すれば、稀な所見も等しく1票を持つ。
この「所見ごとに独立」という性質こそが、本notebookの**所見別重み最適化**という戦略を成立させている。

### 改善点の考察

**他notebookとの比較**（[RSNA Knee DINO Protocol Fusion 0.935](https://www.kaggle.com/code/llccqq624/rsna-knee-dino-protocol-fusion)、
[Head and shoulders, knees and toes 0.936](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes)、
[knee-mri-twelve-findings-from-a-single-model 0.924](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)）:
0.935〜0.936帯が完全に**同一の上流（DINOsaur V10）からのフォーク**で埋まっている。ランタイムを見ると、DINO Protocol Fusion は 2分30秒、
本notebookは 31分45秒。つまり **0.001の差に29分を払っている**。同時に注目すべきは 0.924 の単一モデル notebook で、
アンサンブル無し・TTA無しでこのスコアに到達している。**新規性のある伸びしろは、融合の精緻化ではなく単体モデルの側にある**という強い示唆。

**関連文献**: [OrthoFoundation: A multimodal vision foundation model for generalizable knee pathology (arXiv:2601.18250)](https://arxiv.org/abs/2601.18250)（2026年1月）は、
膝のX線・MRI 120万枚のラベル無し画像で **DINOv3 バックボーンを自己教師あり対照学習で事前学習**した基盤モデル。
本notebookが使っているのは**自然画像で事前学習された汎用DINOv2**なので、ドメイン特化の事前学習に差し替える余地が丸ごと残っている。
また [Large-scale Multi-sequence Pretraining for Generalizable MRI Analysis (arXiv:2508.07165)](https://arxiv.org/pdf/2508.07165) は、
複数シーケンス（T1/T2/PD等）を跨いだMRI特化の事前学習を扱っており、多断面・多シーケンスを持つ本コンペと相性が良い。

**改善提案**

1. **バックボーンをドメイン特化の事前学習に差し替える**（最も期待値が高い）。汎用DINOv2 → 膝画像120万枚で事前学習された OrthoFoundation 系、あるいはMRI多シーケンス事前学習モデルへ。**上位が全員同じ上流からフォークしている今の状況では、バックボーンを変えるだけで誤りの相関が切れ、アンサンブルの多様性としても価値がある**。
2. **ラベル生成そのものを改善する**。現状のソフトラベルは「LLMが1回読んだ結果」。複数LLM・複数プロンプトで読ませてラベルを平均し、**不一致が大きいstudyには学習時のサンプル重みを下げる**（ラベルの不確実性を学習に伝える）。ラベルが全ての律速になっているコンペなので、ここへの投資が最も効く。
3. **58件でのチューニングを止め、cross-fittedな重み推定に切り替える**。1所見あたり平均5件未満での重み推定はほぼノイズ。`bootstrap_shrunk` は正しい方向だが、より根本的には**LLMソフトラベル4,349件を検証に使う**（ラベルは弱いが件数が70倍以上）方が、分散の観点で有利になりうる。
4. **`aggressive` 系プロファイルを最終提出に選ばない**。原著者自身が「これはprivate推定ではなくpublic LBへのプローブだ」と明記している。残り2か月あるので、**今は `raptor065` で情報を取り、終盤で `bootstrap_shrunk` に戻す**という使い分けが妥当。LBプロービングは情報収集としては有効だが、そのまま提出するとshake-upで刺される。
5. **計算コスト対効果を測り直す**。31分45秒 vs 2分30秒で 0.001。この29分を、TTAでも所見別重みでもなく**別バックボーンの推論**に使えば、同じ予算でより大きな多様性が得られる。「効いているから足す」ではなく「**予算あたりで最も効くものに使う**」という発想に切り替えるべき局面。

---

## 今日のまとめ

3本を貫くのは **「アンサンブルの重みは、どれだけのデータで支えられているか」** という問いだった。
S6E8は数万行のOOFを持ちながら `C=0.006` で自由度を殺し、RSNAは58件しかないgold setで所見別に重みを攻め、Biohubはそもそも重みを固定して**設定が動かないこと自体をコードで強制**した。
そして3本とも共通して、上位帯が**同じ上流からのフォークで団子状態**になっている。
このとき小数第4位を追う作業は期待値が低く、効くのは「バックボーンを変える」「ラベル生成を変える」「シードを増やす」といった、**誤りの相関を切る方向の変更**である。
今日の3本で最も実務に持ち帰れるのは、Biohubの Configuration Guard と Pipeline Manifest —
**「configではONだが実体はOFF」というサイレント・デグレードを、人間の注意力ではなくコードで検出する**という発想だ。
