# 2026-08-30 Kaggle日次レビュー

今日の3本は、**「スコアを上げる操作のうち、どれが実力でどれが偶然か」を見分ける** という一点で並んだ。

- **Biohub** は「壊さないための仕組み」を積み上げる方向 — 例外で止める監査、独立モデルによる拒否権、移植ルールの再チューニング。
- **S6E8** は真逆で、**わざと偶然に適合させて見せる**「反面教師」notebook。原著者自身が「これは選ばない」と宣言している。
- **RSNA** は、指標の数式（マクロ平均AUC）を読み込んで **そこで許される自由度だけ**を使い切る方向。

3本を並べると、「指標をよく読む」という同じ行為が、正しく使えば設計指針になり、
踏み外せばLB過剰適合になる、という表裏がはっきり見える。

---

## 1. 【固定枠】Biohub - Cell Tracking During Development — 「Biohub Cell Tracking 92.6%」

- **コンペ**: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)（Research / Code Competition、$60,000、2,860チーム、残り約1か月）
- **notebook**: [Biohub Cell Tracking 92.6%](https://www.kaggle.com/code/salemali7/biohub-cell-tracking-92-6) / 原著者: **SALEM ALI（salemali7）**（COLD氏のnotebookからのfork、+179/−365行）
- **Public Score**: **0.926**（V6、43 votes、Bronze）
- **ファイル**: [`biohub_biohub-cell-tracking-92-6.ipynb`](./biohub_biohub-cell-tracking-92-6.ipynb)

### 学べる主要テクニック

1. **移植ルールの再キャリブレーション** — 他notebook（`kunaldesale2408`）の分裂検出ルール（相互最近傍の姉妹 + t+2での発散テスト）を移植する際、半径を **12/15/10 → 8/11/10 に絞った**。「自分のノード密度の方が高いので、原設定ではFPが14件出る」という理由まで書いてある。コピーで終わらせず、自分の統計に合わせ直す作法。
2. **8視点D4対称TTA へのモンキーパッチ** — 上流の推論スクリプトを実行前に文字列置換し、4方向フリップTTAを回転・転置込みの8視点に拡張。置換に失敗したら必ず警告を出す（`TTA WARNING: block not found`）実装になっている点が重要。
3. **DeepCenter veto（独立検出器による修復の拒否権）** — ギャップクロージングで作った合成ノードを、リンク予測とは別に学習した中心検出UNetに再確認させ、否定されたら取り消す。「繋げばTPが増える」という誘惑に対して、相関の低い2つ目の証拠を要求する。
4. **非対称なILP重み** — `DISAPPEARANCE_WEIGHT=1.5` / `APPEARANCE_WEIGHT=0.0`。「途中で消える」を高コスト、「途中から現れる」を無コストにして、エッジFNを潰す方向に非対称化している。
5. **フェイルクローズドな提出監査** — 最終セルで列名・行IDの連番・row_type・データセット網羅性を全部チェックし、1つでも違反したら `RuntimeError`。ラベル不要で書けるのに、5時間のGPU推論を無駄にする事故を構造的に防ぐ。
6. **厳格なno-op設計** — 診断用ヘルパーを2つ定義しているが、呼び出しは両方コメントアウト・例外も握りつぶす。「診断を足したらスコアが変わった」を起こさない書き方。

### 評価指標の要約

**`score = adjusted_edge_jaccard + 0.1 × division_jaccard`**。ノードは時点ごとに **7.0µmゲート付きの最適二部マッチング**（物理スケール z=1.625, y=x=0.40625 µm/voxel）で対応付け、
エッジは「両端がGTノードにマッチし、かつGT側にも同じエッジがある」ときだけTP。正解は**疎ラベル**なので「ノードを大量に出せば得」を防ぐ **node-count penalty** が入っており、その副作用でスコアが1.0を超えることもある。
分裂は稀イベントでエッジ数に埋もれるため、別枠10%で保護されている。
このnotebookは、高い検出閾値（0.96875）＋短トラック除去でnode-count penaltyを避け、非対称ILP重みでエッジFNを潰し、移植ルールで `div_j` を0→0.0625に引き上げる（総合 +0.0046）という、**3つの指標成分それぞれに手を打った**構成になっている。

### 改善点の考察

**他notebookとの比較**（同コンペ上位帯を確認: [Biohub Harmonic Fusion 0.928](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion)、[Biohub 0.928 LB](https://www.kaggle.com/code/evgendvorkin/biohub-0-928-lb)、[biohub-sdw60 0.927](https://www.kaggle.com/code/arnav170/biohub-sdw60)、[Biohub medal v1 0.916](https://www.kaggle.com/code/backtracking/biohub-medal-v1)）:
上位帯は依然 **0.926〜0.931 に密集**し、構成要素（UNet-Transformer検出 + ILP + 後処理）は共通。
このnotebookが**採用していない**主な手法は、(a) 双方向エッジ確率の調和平均融合（`2ab/(a+b)`、0.928帯が使用）、(b) Configuration Guard による設定ドリフト検出、(c) 3seed以上のマルチシード。
逆にこのnotebookだけが持つのは **分裂ルールの移植と再チューニング**で、`div_j` に手を付けた数少ない公開notebookである。

**関連文献**: [ARGUS: Accelerated, Robust, General, and Unsupervised Cell Tracking Solutions (arXiv:2607.08297)](https://arxiv.org/html/2607.08297) は、**教師なし**で細胞追跡を行う方向性を示しており、注釈が疎な本コンペのデータ特性と相性がよい。
また [Higher-Order Cell Tracking Transformer (arXiv:2607.11754)](https://arxiv.org/abs/2607.11754) は「エッジ同士を相互にattentionさせる」edge-centric な設計で、分裂が系譜を絡ませる問題に正面から取り組んでいる。

**改善提案**

1. **動画ごとの動的検出閾値**（最も費用対効果が高い）。現在 `DET_THRESHOLD=0.96875` は全動画一律だが、`estimated_number_of_nodes` は動画ごとに与えられている。予測ノード数が推定値に一致するよう**二分探索で閾値を決める**だけで、node-count penaltyを動画単位で最適化できる。実装は20行程度。
2. **分裂検出への時間的事前知識の導入**。`SAFE_DIV_*` の条件は現状**すべて空間距離**で、細胞周期の典型長という情報が丸ごと未使用。「直前に分裂した細胞が2〜3フレーム後に再分裂する」候補にペナルティを掛ければ、`div_j` のFPをさらに削れる。分裂枠は10%あるので、`div_j` を 0.0625→0.15 に上げるだけで総合 +0.009 相当。
3. **双方向エッジ融合を追加する**。0.928帯が使っている `p_fwd`/`p_rev` の調和平均融合をこのパイプラインに乗せる。「片思いリンク」を落とす効果はこのnotebookの高閾値戦略と方向性が一致しており、競合しない。
4. **veto を分裂判定にも掛ける**。現在 `DEEPCENTER_SAFE_DIV_VETO=0`（無効）になっている。せっかく独立検出器があるのに、最も誤りやすい分裂の判定には使っていない。まず有効化して効果を測るのが素直。
5. **移植ルールの半径を、密度に応じて動的化する**。8/11/10 という値は「自分のノード密度」に対する定数だが、密度は動画内でも領域によって変わる。既に `frame_local_spacing`（局所間隔の計測）が実装済みなので、ギャップクロージングと同じ密度適応の仕組みを分裂半径にも流用できる。追加コストがほぼゼロ。

---

## 2. 【Playground】Playground Series S6E8 — 「🚨 OVERFITTING TRAP - Do Not Copy」

- **コンペ**: [Predicting Smartphone Addiction (S6E8)](https://www.kaggle.com/competitions/playground-series-s6e8)（3,273チーム、**残り2日**）
- **notebook**: [🚨 OVERFITTING TRAP - Do Not Copy](https://www.kaggle.com/code/najiama/overfitting-trap-do-not-copy) / 原著者: **NAJI（najiama）**
- **Public Score**: **0.97129**（V6、28 votes、Bronze）— Rank #22相当、公開notebookの最前線
- **ファイル**: [`playground_overfitting-trap-do-not-copy.ipynb`](./playground_overfitting-trap-do-not-copy.ipynb)

### 学べる主要テクニック（＝「やってはいけない」の実演）

1. **Reverse Micro-Sorting** — ベースラインの順位を `pd.qcut(q=500)` で500バケットに分け、`np.lexsort((id, -lgbm_rank, bucket))` で**バケット内だけ**を自前LGBMの順位の**符号を反転して**並べ替える。大域順位は保たれるのでAUCの本体は壊れず、微細な順位だけがPublic LBの20%サンプルに適合する。+0.00001 の「改善」。
2. **logitブレンドが壊れる条件** — 相手の提出ファイルが既に**厳密ランク化済み**（1/N, 2/N, ...）だと、両端が0/1に極めて近いため `logit` が±∞近くに発散し、数十件が全体を支配する。0.97100 → 0.97093 に低下した。**混ぜる前に分布を見る**。
3. **不確実性マスキング（Uncertainty Masking）** — `mask = (rank>0.2)&(rank<0.8)` で「ベースラインが迷っている中間60%」にだけ弱いモデルを注入する。狙い自体は真っ当で実務でも使えるが、KNN（OOF AUC 0.957）がベースライン（0.971帯）に対して弱すぎて効かなかった。
4. **負の重みブレンド** — 弱いモデルを**引き算**してバイアスを打ち消す。OOF検証なしにやると純粋なLB過剰適合になる典型。
5. **タイブレーカとしての微小重み** — `w=0.001` は値をほぼ動かさないが、同点の並び順だけを変える。AUCは同点の扱いで僅かに動く。

### 評価指標の要約

**ROC-AUC**（二値分類）。「無作為な陽性1件が、無作為な陰性1件より高いスコアを得る確率」に等しく、**順位だけ**で決まるため単調変換に不変。
だから上位提出は軒並み厳密ランク形式になっている。不均衡データでaccuracyが機能しない状況で、閾値を固定せずに順位づけ能力を測れるのが採用理由。
このnotebookは**その順位不変性を逆手に取り**、Public LB（テストの約20%）のノイズにだけ適合する微細な順位操作を行っている。

### 改善点の考察

**他notebookとの比較**（同コンペの0.9712帯を確認: [S6E8: Elite Rank Average Ensemble 0.97126](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97126)、[[S6E8] Top 20 Formula: Dual Master Rank Blend 0.97128](https://www.kaggle.com/code/souvikdbiswas/s6e8-top-20-formula-dual-master-rank-blend)、Pure LGBM Model LB 0.96999 / CV 0.96881）:
公開notebookのスコアは **0.97099〜0.97129 の 0.0003 幅に20本以上が密集**している。
テスト30万行に対しこの差は数十サンプルの順位入れ替えに相当し、**統計的にはほぼ区別できない**。
なおこのnotebookが採用していないのは、Nested CV・OOFに基づく重み最適化・単調性制約・敵対的検証といった「順位を実力で上げる」手法すべてである（それが主題だから当然だが）。

**関連文献**: [The Ladder: A Reliable Leaderboard for Machine Learning Competitions (Blum & Hardt, ICML 2015)](https://proceedings.mlr.press/v37/blum15.pdf) は、
「提出のたびにLBスコアを見て次を決める」という適応的な手続きが、なぜ統計的保証を壊すのか（adaptive data analysis）を定式化し、
**前回より有意に良いときだけスコアを更新する**Ladderメカニズムで過剰適合を抑えられることを示している。
このnotebookがやっている「+0.00001を追う」行為は、まさにLadderが弾くために設計されたパターンそのもの。

**改善提案**

1. **同じmicro-sortingを、符号を正のまま OOF で検証して使う**。「大域順位は強いモデル、局所順位は別モデル」という構造自体は正当。OOFでバケット数と重みを選び、符号を反転させたくなった時点で「これはノイズだ」と判定するルールにすればよい。
2. **バケット数をOOFで選ぶ**。500という値はPublic LBを見て決めたもの。OOF上でバケット数を 50 / 200 / 500 / 2000 と振れば、「局所順位に信号がどれだけ残っているか」が測れる。信号がゼロなら、どの粒度でも改善しないはず。
3. **ブートストラップ信頼区間を出す**。0.97100 と 0.97101 の差が有意かを、OOF予測のブートストラップ再標本化でAUCの分布として示す。ほぼ確実に区間が重なり、**差が測定不能である**ことが定量的に言える。この1枚のヒストグラムが、この notebook 全体の主張の証明になる。
4. **CVとLBの相関そのものを記録する**。原著者は「CV 0.97035 / LB 0.97126 の別モデルを選ぶ」と書いているが、そのCV-LB散布図があれば説得力が跳ね上がる。手元の全実験について (CV, LB) を1つのCSVに貯めておく習慣は、どのコンペでも効く。
5. **shakeup後の答え合わせを残す**（残り2日）。このnotebookはPrivate LB公開後に「予測が当たったか」を検証できる稀な教材。Private公開後にスコアがどれだけ落ちたかを追記すれば、教材としての価値が完成する。

---

## 3. 【開催中の実コンペ】RSNA Knee Abnormality Detection — 「RSNA Knee DINO Protocol Fusion」

- **コンペ**: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)（Research / Code Competition、$77,000、2,663チーム、残り約2か月）
- **notebook**: [RSNA Knee DINO Protocol Fusion](https://www.kaggle.com/code/llccqq624/rsna-knee-dino-protocol-fusion) / 原著者: **JIACHEN LI（llccqq624）**
- **Public Score**: **0.935**（V16、24 votes、Bronze）
- **ファイル**: [`competition_rsna-knee-dino-protocol-fusion.ipynb`](./competition_rsna-knee-dino-protocol-fusion.ipynb)

### 学べる主要テクニック

1. **4つの独立メンバーによる逐次ブレンド** — 巨大コードセル4つが、それぞれ (1) DINOv2 frontier再現、(2) マルチスロットViT、(3) RadImageNet ResNet50 + 88特徴量transformerキャリブレーション、(4) CoAtNet単体、という別々のモデル。セルを進むごとに `submission.csv` が上書きされていく。**事前学習データが4系統とも違う**ので、誤りが相関しにくい。
2. **Protocol Fusion（撮像プロトコル別スロット）** — `SLOTS = [(Sagittal,1),(Sagittal,0),(Coronal,1),(Coronal,0),(Axial,1),(Axial,0)]` の6スロットを明示的に分けて入力し、**欠けているスロットは他で代用せずマスクする**。「Axialが無いからSagittalで埋める」をやらない。
3. **DICOM幾何情報からの laterality 復元** — タグを鵜呑みにせず、患者座標系での位置から左右を判定し直す。左右を間違えると「内側OA / 外側OA / 内側半月板 / 外側半月板」の4所見が丸ごと入れ替わるので、マクロ平均AUCへの影響が甚大。
4. **物理単位でのクロップ（`CROP_MM = 130.0`）** — ピクセル数ではなくmmで切る。22か国以上の施設からデータが来ており、FOVもピクセル間隔もバラバラなので、この正規化なしには施設間で入力の意味が変わってしまう。
5. **所見ごとに異なるブレンド重み（`_coatnet_weight[label]`）** — マクロ平均AUCは12本の独立したAUCの平均なので、所見ごとに重みを最適化しても互いに悪影響が出ない。**指標の数式から許される自由度を使い切る**発想。
6. **フェイルオープンなキャリブレーション** — `try/except` でキャリブレーションが失敗したら生の出力を保持して続行。提出形式の検証は「壊れていたら止める」（フェイルクローズド）なのに対し、任意の改善処理は「失敗したらスキップ」と、**止める/続けるを目的別に使い分けている**。

### 評価指標の要約

**マクロ平均 ROC-AUC**（12所見のAUCの単純平均）。所見ごとに有病率が大きく違い（ACL断裂や変形性関節症は多い一方、骨折やベーカー嚢腫は稀）、
マイクロ平均だと高頻度の所見だけでスコアが決まってしまうため、**12所見を等しく1/12ずつ扱う**マクロ平均が採用されている。
結果として「稀な所見での性能」が正面から効く。AUCの順位不変性から、このnotebookは全ての中間出力を列ごとに `rank(pct=True)` に直してから混ぜており、
さらに `_RAD_EXCLUDE = ("Baker's", "Fracture")` として**陽性数の少ない2所見はキャリブレーションの対象外**にしている（1所見の悪化がそのまま 1/12 効くため、「触らない」が正解になる場面）。

### 改善点の考察

**他notebookとの比較**（上位帯を確認: [Head and shoulders, knees and toes 0.936](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes)、[rsna-base 0.936](https://www.kaggle.com/code/anvithpothula/rsna-base)、[RSNA Knee | DINOsaur V4 0.936](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4)、[Knee MRI: twelve findings from a single model 0.924](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)）:
公開帯のトップは 0.935〜0.936 に貼り付いており、**その全員が同じ公開再現アセットを土台にしている**。
一方 `dreaddevelopment` の「twelve findings from a single model」は **アンサンブルもTTAも使わず単体で0.924** を出しており、
これは4メンバー融合との差がわずか0.011しかないことを意味する。**融合による上積みは既に逓減している**。
このnotebookが採用していないのは、(a) **レポートテキストの直接利用**（マルチモーダル性そのもの）、(b) 所見間の共起事前知識による補正、(c) 効率トラックを意識した軽量化。

**関連文献**: RSNAの公式解説によれば、このデータセットは **4,407 studyのうち構造化ラベルが付いているのは58件だけ**で、
残りは**12言語の放射線科レポート**から弱教師ラベルを作る必要がある（[RSNA AI Challenge 公式](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)、[AuntMinnie 報道](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15831808/rsna-launches-2026-knee-abnormality-detection-ai-challenge)）。
つまり**このコンペの本質的なボトルネックは画像モデルではなくラベル生成の質**である。
関連して [Multi-task weak supervision enables anatomically-resolved abnormality detection (PMC7994797)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7994797/) は、
レポートから抽出した不完全なラベルをマルチタスクで統合し、解剖学的に分解された異常検出を実現する枠組みを示している。

**改善提案**

1. **レポート由来ラベルの質を上げる方が、画像モデルを増やすより効く**。公開帯が0.935で飽和しているのは、全員が同じ弱教師ラベルを使っているからと考えられる。多言語の否定表現（「ACL断裂なし」「no evidence of tear」）の検出精度を上げるだけで、稀な所見のラベルノイズが減り、マクロ平均が直接改善する可能性が高い。**最も伸びしろが大きい提案**。
2. **所見間の共起構造を後処理に入れる**。「内側半月板断裂 + 内側OA」「関節液貯留 + 滑膜炎」は臨床的に強く共起する。12所見を独立に予測したあと、共起行列に基づく低ランク補正を掛ける（順位を保つ範囲で）。マクロ平均AUCは所見ごとに測るので、**稀な所見の順位を、頻出所見の情報で補強できる**のが利点。
3. **効率トラックを狙う**。本コンペには別枠で **Efficiency Prize（$18,000）** があり、`Efficiency = AUC項 + RuntimeSeconds/32400` を最小化する。4メンバー融合は精度では0.935でも実行時間が長く、効率スコアでは不利。単体0.924のCoAtNetは、**精度を0.011落とすだけで実行時間を大幅に削れる**ので、効率トラックでは有力候補になりうる。競争が薄い側を狙う戦略として現実的。
4. **稀な所見に特化したヘッドを別枠で学習する**。骨折・ベーカー嚢腫は現状「キャリブレーション対象外」として放置されている。12所見を1つのヘッドで出すのではなく、稀な2〜3所見だけを LDAM-DRW などの不均衡対策付きで別途学習し、その所見だけ差し替える。マクロ平均では稀所見の改善が満額（1/12ずつ）反映される。
5. **スロット欠損パターンごとの性能を分解して見る**。「Axialが無いstudy」「Coronalしか無いstudy」で所見別AUCがどう変わるかを分解すれば、どのスロットがどの所見を支えているかが分かる。欠損パターンがテスト側で偏っていた場合のリスク評価にもなり、`_pick_series_for_slot` の改善方針が定量的に決まる。

---

## 今日のまとめ

「指標をよく読む」は今日の3本に共通する行為だが、行き先が正反対だったのが面白い。
RSNAは**マクロ平均AUCが所見ごとに独立だから所見ごとに重みを変えてよい**という、数式から導かれる正当な自由度を使い切っていた。
S6E8は**AUCが順位だけで決まるから局所順位を弄れば数字が動く**という、同じ性質を悪用して +0.00001 を得ていた。
両者を分けているのは指標の理解度ではなく、**「その操作をOOFで検証したか」の一点**だけである。

Biohubの「例外で止める監査」「独立モデルによる拒否権」「移植ルールの再チューニング」は、
どれも「自分の思い込みに対して、別の証拠を要求する」という同じ形をしている。
LB過剰適合とは、要するに**その要求を止めてしまった状態**のことだ。

---

## Sources

- [Biohub - Cell Tracking During Development（評価指標）](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)
- [RSNA Knee Abnormality Detection（評価指標）](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
- [ARGUS: Accelerated, Robust, General, and Unsupervised Cell Tracking Solutions (arXiv:2607.08297)](https://arxiv.org/html/2607.08297)
- [Higher-Order Cell Tracking Transformer (arXiv:2607.11754)](https://arxiv.org/abs/2607.11754)
- [The Ladder: A Reliable Leaderboard for Machine Learning Competitions (Blum & Hardt, ICML 2015)](https://proceedings.mlr.press/v37/blum15.pdf)
- [RSNA Knee MRI AI Challenge 公式ページ](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [RSNA launches 2026 Knee Abnormality Detection AI Challenge（AuntMinnie）](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15831808/rsna-launches-2026-knee-abnormality-detection-ai-challenge)
- [Multi-task weak supervision enables anatomically-resolved abnormality detection (PMC7994797)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7994797/)
