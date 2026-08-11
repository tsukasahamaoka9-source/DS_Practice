# 2026-08-12 Paper Digest

## Cell-TRACTR: A transformer-based model for end-to-end segmentation and tracking of cells

- 著者: Owen M. O'Connor, Mary J. Dunlop（Boston University）
- 発表: PLOS Computational Biology, 2025年5月23日（Vol.21, Issue 5, e1013071）
- 分野: 細胞追跡・コンピュータビジョン（Transformerベースのend-to-end検出+追跡）
- リンク: https://doi.org/10.1371/journal.pcbi.1013071
- コード: https://gitlab.com/dunloplab/Cell-TRACTR （評価指標Cell-HOTAのコードは https://gitlab.com/dunloplab/Cell-HOTA ）

直近のKaggle日次レビュー（2026-08-10, 2026-08-11）で扱ったBiohub - Cell Tracking During Developmentの上位notebook群は、いずれも「検出→（ILP等で）リンク→後処理」という3段階のtracking-by-detection方式を取っている。2026-08-11のレビューで改善案として挙げた「検出とリンク予測を単一モデルで同時学習するend-to-end方向」の具体例として、本論文を選んだ。2026-08-05に扱ったHigher-Order Cell Tracking Transformer（HOCT、エッジ中心アテンションで非同質性グラフ問題に対処）とは異なるアプローチで、こちらは「グラフを作ってスコアリングする」のではなく「検出とトラッキングをそもそも1つのTransformerで一体化する」方向性を取る。

### 背景・課題

細胞追跡（時系列顕微鏡画像から個々の細胞の系譜・分裂イベントを復元するタスク）は、頻繁な分裂イベント・見た目が酷似した多数の物体・低フレームレートによる非線形な動きなど、一般的な物体追跡（歩行者・車両など）にはない難しさを持つ。既存のCNNベース手法（U-Net系）は空間的・大域的な文脈依存性の扱いに限界があり、また多くの手法は検出とトラッキングを別々のステージ・別々のデコーダで行うか、後処理（非最大抑制など）に依存しており、パイプラインが複雑になりがちだった。

著者らは、似た者同士（ダンサー）を追跡するMOT（Multi-Object Tracking）データセットDanceTrackでSOTAを達成したMOTRv3に着想を得た。ダンサーの見た目の類似性・多様な動きのパターンは細胞追跡の性質と似ており、MOTRv3系のアーキテクチャが細胞追跡にも有効だろうという仮説から本研究が始まっている。

### 提案手法の概要

- **DETRベースのend-to-end構成**: CNN（ResNet）バックボーンで画像特徴を抽出し、Deformable Attentionを使うTransformerエンコーダで多スケール特徴を精製、query selection（Deformable DETRの"two-stage"方式）で物体クエリ（object query）を初期化する。デコーダで物体クエリが特徴に注意（cross-attention）し、クラス（cell/no object）・バウンディングボックス・セグメンテーションマスクを出力する。非最大抑制のような後処理は一切不要。
- **トラッキング（TrackFormer/MOTR方式）**: 最初のフレームでは物体クエリで全細胞を検出。以降のフレームでは、前フレームで検出済みの細胞の出力埋め込みが「トラッククエリ（track query）」としてそのまま次フレームに引き継がれる。トラッククエリと（新規細胞検出用の）物体クエリはデコーダ内でself-attentionにより互いを認識し合い、同じ細胞を物体クエリとトラッククエリが二重検出しないよう調整される。
- **分裂イベントの扱い**: 各出力埋め込みは常に2つの予測（自分自身＋分裂した場合の娘細胞候補）を出す。両方が"cell"と分類されれば分裂と判定する、というシンプルな設計で、特別な分裂検出モジュールを別途持たない。
- **Cell-HOTA指標の提案**: MOTコミュニティで標準化しつつあるHOTA（Higher Order Tracking Accuracy：検出精度DetAと関連付け精度AssAをバランスよく評価する指標）を、細胞分裂を評価できるように拡張。分裂精度DivAを新設し、AssAとDivAの幾何平均（AssDivA）とDetAをさらに幾何平均してCell-HOTAとする。CTC（Cell Tracking Challenge）で従来使われてきたOPCTB指標（SEGとTRAの平均）は、①分裂タイミングのわずかなズレ（1フレーム早い/遅い）を過度に厳しく罰する、②セグメンテーション精度（SEG）に評価が偏りトラッキング精度の違いを反映しにくい、という2つの弱点が指摘されている。Cell-HOTAでは分裂タイミングが正解の前後1フレーム以内なら許容する「flexible division」オプションを導入し、細胞分裂タイミングの正解自体が本質的に主観的（asubjective）であるという問題に対処している。

### 主な結果

- 細菌（E. coliのmother machine、1方向成長）とマウス由来培養細胞（DeepCellのDynamicNuclearNet、2次元自由成長）の2種類のデータセットで、DeLTA・EmbedTrack・Trackastra・Caliban（DeepCell専用に調整済み）と比較。
- Cell-HOTA0.5（IoU閾値0.5での評価）で、細菌データではCell-TRACTRが最高スコア。特にAssA（トラッキング精度）とDivA（分裂精度）でTrackastra・EmbedTrack・DeLTAを上回った一方、DetA（検出精度）はEmbedTrackがわずかに上回った。
- マウス細胞データでは、DeepCell専用にチューニングされたCalibanが総合最高だが、Cell-TRACTRはAssAで最高スコアを記録し、他の指標でも遜色ない性能。
- 「Flexible division」を無効化する（分裂タイミングのズレを一切許容しない）と、全モデルでDivAが大きく低下——分裂タイミングの厳密一致がいかに全モデル共通の弱点かを可視化している。
- セグメンテーション＋トラッキングを両方行うモデルの中では、Cell-TRACTRの推論速度（FPS）が最速だった。

### Kaggleでの実践への示唆

Biohubコンペの上位notebook（0.915クラスタ）は、検出→候補グラフ生成→ILP最適化→後処理（ギャップ補完・加速度整合性ボーナス等）という多段パイプラインを手作業でチューニングする方向に進んでいる。本論文は「そもそも検出とリンクを1つのTransformerで統合し、後処理を丸ごと不要にする」という別解を示しており、直接移植は大工事だが、部分的に借用できる発想が2つある。

1つ目は分裂判定の設計。現行notebookのILP後処理は分裂ジオメトリチェックを個別のルールとして実装しているが、Cell-TRACTRの「同一の起点候補から2つの高信頼マッチが同時に立つ場合を分裂とみなす」という単純な二値判定は、既存のリンクスコアリング後処理に低コストで足せる可能性がある（新しいネットワークの学習は不要で、候補リンクの集計ロジックだけの変更で試せる）。

2つ目は評価指標そのものへの示唆。Biohubコンペで観測されている「144チームが0.915付近に集中」という現象は、本論文が指摘するOPCTB系指標の性質（SEGへの過度な偏り、分裂タイミングのわずかなズレへの過剰なペナルティ）と整合的である可能性がある。ローカル評価にCell-HOTA相当の「分裂タイミング±1フレーム許容」の緩和ルールを実装し、node_recall/edge_jaccardを検出・関連付け・分裂の3軸に分解して計測すれば、現在の手法がどの軸で頭打ちになっているかを特定でき、次の有意な改善（0.916超え）に繋がる弱点発見に使えると考えられる。

### 実装可能な仮説

1. **手法**: 現行のILP後処理に、Cell-TRACTRの分裂判定ロジック（起点候補から高信頼マッチが2本同時に立つ場合を分裂とみなす）を追加特徴量として組み込む。 **期待インパクト**: 中（分裂点でのidentity swap削減、edge_jaccardの改善）。 **工数**: 低〜中（新規学習不要、候補リンク集計ロジックの追加のみ）。 **出典**: Cell-TRACTR論文 Results節「Cell-TRACTR architecture: Cell division」。
2. **手法**: ローカル評価パイプラインに、Cell-HOTA相当の「分裂タイミング±1フレーム許容」を実装し、node_recall/edge_jaccardを検出・関連付け・分裂の3軸に分解して測定する。 **期待インパクト**: 高（0.915クラスタ停滞の原因軸を特定でき、次の改善の優先順位付けに直結）。 **工数**: 低（評価コードのみ、モデル変更不要）。 **出典**: Cell-TRACTR論文 Fig.4D（flexible divisionを無効化すると全モデルでDivAが大きく低下する結果）。
3. **手法**: 候補リンクの再スコアリング特徴量に、前フレーム検出の埋め込みベクトルとの類似度（track queryの継続性に相当する特徴）を追加する。既存の「ローカル関連性ランカー」（22特徴量）の拡張として実装可能。 **期待インパクト**: 中（既存特徴量では捉えにくい大域的な系譜の一貫性を補強）。 **工数**: 中〜高（埋め込み抽出用の軽量モデルの追加学習が必要）。 **出典**: Cell-TRACTR論文 Fig.2（track queryによるフレーム間の一貫性維持の仕組み）。

### 備考

本要約はPLOS Computational Biology掲載の論文本文（Abstract〜Results前半、Discussion手前まで）をブラウザ経由で読み込んで作成した。Methods節・Discussion節の詳細（損失関数の具体的な数式、ハイパーパラメータ、限界点の議論など）までは読み込めておらず、必要であれば原文を参照されたい。
