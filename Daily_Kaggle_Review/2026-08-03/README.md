# 2026-08-03 Kaggle Daily Review

今日は固定枠のBiohub、Playground Series（新シーズンS6E8）、開催中の実コンペ（金融・Hull Tactical）の3本を解説付きノートブック化しました。

## 1. Biohub: Cell Tracking: Learned Graph w Gap Recovery

- notebook: `biohub_biohub-cell-tracking-learned-graph-w-gap-recovery.ipynb`
- 元notebook: [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery) by Pilkwang Kim（184 upvotes, Gold, Public Score 0.893 / Best Score 0.894）
- コンペ: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)（ゼブラフィッシュ胚発生の3D顕微鏡動画から細胞を検出・追跡、賞金$60,000）

学べる主要テクニック:
- `TemporalUNet3D`による学習ベースの細胞中心検出と、ノード間クロスアテンションtransformerによる学習ベースのエッジ（リンク）スコアリング
- ILP（整数線形計画法）によるグラフ全体の大域最適化で、局所的な矛盾（1ノードに複数の親等）を排除
- Hungarian法（`linear_sum_assignment`）を使った厳密な最小コストマッチングによるgap close・motion relink
- 絶対数キャップと割合キャップを併用した、動画サイズに頑健な後処理パラメータ設計
- サブピクセル精度の中心精緻化（画素強度を重みとした重心計算）

評価指標の要約: score = adjusted_edge_jaccard + 0.1×division_jaccard。ノード同士の対応（エッジ）精度をJaccard係数で測り、細胞分裂の検出精度を0.1倍の重みで加える設計。今回のnotebookはエッジ予測を学習ベースで行いつつ、後処理（gap close・division復元）の閾値を保守的に絞った「高recall・低over-repair」プロファイル（V32, "Recall With Cleaner Repair"）。

改善点の考察:
- Codeタブを見ると、このコンペには `Metric_hack_last_call` のような露骨な命名のnotebook群（スコア0.95〜0.97台に集中）が多数存在し、評価指標の境界条件を突いた「メトリックハック」が疑われる。今回選定したnotebookはこのクラスタとは異なる正攻法のスコア（0.894）であり、著者自身も「学習ベースの手法はこの時点で一区切り」と明言しているが、次にモデリングの主戦場を移す予定と述べている。ハックされたスコアを鵜呑みにせず、正攻法のアプローチから何を学べるかを重視すべき。
- 著者コメントによれば、このバージョンは「gap追加やdivision復元を絞ることで、残っている損失が本当にover-repairによるものか」を検証する実験的なプロファイルであり、まだ結論は出ていない。次の実験として、gap2やsafe_divisionのパラメータをさらに系統的にグリッドサーチし、どの閾値が実際にスコアに寄与しているかを定量化する余地がある。
- 検出・エッジ予測モデル自体（`TemporalUNet3D`、cross-attention transformer）の層構造は外部リポジトリに切り出されており、notebook単体からは学習の詳細（データ拡張、損失の重み付けの妥当性検証等）が読み取れない。オーケストレーション層（このnotebook）と学習コードが分離されているのは可読性の面で良い設計だが、再現性検証の観点では両方公開されているとより良い。
- 関連文献では、細胞追跡分野は"Higher-Order Cell Tracking Transformer"（候補リンク同士が3D幾何制約のもとで互いにattendするedge-centricなアーキテクチャ、[arXiv:2607.11754](https://arxiv.org/abs/2607.11754)）のような、エッジ同士の高次関係（細胞分裂で系譜が分岐する際のノード埋め込み空間での絡み合いなど）を明示的にモデル化する方向へ進化している。また"How To Make Your Cell Tracker Say 'I dunno!'"（[arXiv:2503.09244](https://arxiv.org/pdf/2503.09244)）のように、追跡の不確実性を明示的に定量化する研究も出てきており、今回のnotebookのような決定的なルールベース後処理に対して、確信度に応じた後処理の強弱づけを組み込む余地がある。

## 2. Playground: Predicting Smartphone Addiction (S6E8) — Hill Climbing Ensemble

- notebook: `playground_hill-climbing-ensemble-for-smartphone-addiction.ipynb`
- 元notebook: [Hill Climbing Ensemble for Smartphone Addiction](https://www.kaggle.com/code/omidbaghchehsaraei/hill-climbing-ensemble-for-smartphone-addiction) by Omid Baghcheh Saraei（19 upvotes, Bronze, Public/Best Score 0.97005）
- コンペ: [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)（Playground Series Season 6 Episode 8。新シーズン開始。スマートフォン依存度リスクの予測）

学べる主要テクニック:
- XGBoost・LightGBM・CatBoost・TabM・TabNet・FT-Transformer・ResNetという、決定木系＋複数の深層学習表形式アーキテクチャのOOF予測を集めたアンサンブル
- Hill Climbing（貪欲法による逐次モデル追加＋重み探索）による、単体最強モデルを上回るアンサンブル構築
- 負の重み（`negative_weights=True`）を許容し、「予測を打ち消す」形の寄与も探索対象にする設計
- 本番評価指標（ROC-AUC）をアンサンブル最適化の目的関数に直接使う

評価指標の要約: ROC-AUC（正例を負例より高くスコアリングできる確率）。Hill Climbingの目的関数にもこの指標を直接設定しており、代理指標を経由しない直接最適化になっている。

改善点の考察:
- 同コンペのCodeタブ上位には「S6E8 honest OOF blend 54 models」（Score 0.97061）や「PlaygroundS6E8|Public|L2Stack|V1」（2段スタッキング、Score 0.97021）が存在し、今回選んだ6モデルのHill Climbing（0.97005）よりわずかに高いスコアを出している。特にL2スタッキング（メタモデルによる非線形な組み合わせ学習）は、Hill Climbingの線形重み付けでは捉えられない複雑な相互作用を学習できる可能性がある。
- 今回のnotebookはベースモデルを別notebookで個別学習した「OOF/testの読み込み」に依存しているため、ベースモデル自体のハイパーパラメータチューニングの余地は検証できていない。個々のモデル単体スコア（RealMLP 0.96844〜ResNet 0.96610）の差は小さくないため、弱いモデル（ResNet, FT-Transformer）自体のチューニングを改善すればアンサンブル全体の底上げにもつながる可能性がある。
- Hill Climbingはprecision=0.001の粒度でしか重みを探索しないため、真の最適解からわずかにずれる可能性がある。関連文献によれば、貪欲法（Hill Climbing/greedy selection）とスタッキングは統計的に有意な性能差があり、集約スタッキング（aggregated stacking）の方が一般的に性能が高いとの比較研究がある（[mlr-org: Greedy Ensemble Selection and Stacking](https://mlr-org.com/gallery/appliedml/2025-05-06-ensembles-stacking-ges-stacking/)）。次のステップとしてHill Climbingで得た重みを初期値としたスタッキングメタモデルとの比較実験が考えられる。
- 実際に2025年のKaggle Playground S5E12優勝解法でも「Hill Climbing + Ridge Ensemble」（[1st Place Solution](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)）が採用されており、Hill Climbingの後にRidge回帰などの線形メタモデルを重ねる二段構えも有力な改善方向。

## 3. 実コンペ: Hull Tactical - Market Prediction — Ensembling Trees with Online Training

- notebook: `competition_hull-tactical-ensembling-trees-online-training-001.ipynb`
- 元notebook: [Hull Tactical-Ensembling-trees-Online-Training-001](https://www.kaggle.com/code/youneseloiarm/hull-tactical-ensembling-trees-online-training-001) by El Younes（132 upvotes, Gold, Private Score 0.388）
- コンペ: [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)（匿名化された市場特徴量からS&P500の翌日超過リターンを予測、賞金$100,000。訓練フェーズ終了後も2026年6月26日まで実市場データに対するforecasting phaseが継続中）

学べる主要テクニック:
- XGBoost・LightGBM・CatBoostの3モデルアンサンブル
- **Kaggle評価APIの推論ステップ内でオンライン学習（新しい行が届くたびに再学習）を行う**、時系列コンペ特有の実装パターン
- オンライン学習の検証データは「直近かつごく短いウィンドウ」に限定すべきという、時系列特有の落とし穴の明示的な警告（validation_split=0.25は9年分の検証を意味してしまうため誤り、0.0001〜0.003程度が妥当という具体的な指摘）
- 目的変数のラグ特徴量化によるリーク防止
- 予測値を増幅・クリップしてポジションサイズに変換する後処理設計

評価指標の要約: 市場平均に対する幾何平均Sharpe比を、ボラティリティ超過（市場の1.2倍超）へのペナルティとリターン未達への二次ペナルティで割った独自指標。このnotebookは指標を直接最適化してはいないが、後処理のポジションサイズ変換式（`allocation = 1.0 + 50 * pred`）の倍率をローカルでこの指標に対して評価しながら調整しており（0.046→1.066へ改善）、間接的な指標整合を行っている。なお同コンペのCodeタブには評価指標の境界を突いたと思われるスコア17.396前後に集中したnotebook群が多数あり、今回選んだnotebookはそのクラスタとは異なる地に足のついた値。

改善点の考察:
- **最大の学び**: オンライン学習における`validation_split`の設定ミスという、一般的な機械学習の直感（検証データは多いほど安心）が時系列のオンライン学習では逆効果になる典型例を著者自身が明示的に警告している。この教訓は今日扱ったPlaygroundやBiohubのnotebookには無い、実コンペならではの実務的な価値。
- ポジションサイズへの変換係数（×50）が経験的に選ばれており、理論的な導出（例えばKelly基準のようなポジションサイジング理論との対応）が示されていない。関連文献では、金融時系列における概念ドリフト対応としてストリーミング勾配ブースティング（[Streaming Gradient Boosting for Evolving Data Streams](https://lamarr-institute.org/blog/streaming-gradient-boosting/)）や、複数モデルをオンラインでアンサンブルするOneNet（[arXiv:2309.12659](https://arxiv.org/pdf/2309.12659)）のような手法が提案されており、単純な線形変換より理論的に裏付けられたポジションサイジング手法を導入する余地がある。
- オンライン学習は1行あたり約27秒かかり、1年分で最大2時間という計算コストが発生する。著者はこれを許容範囲としているが、Code Competitionの実行時間制限（GPU 8時間）に対してどれだけ余裕があるかの定量的な議論は無い。将来的にテストセットが拡張される「forecasting phase」でこのコストがボトルネックにならないか検証する価値がある。
- モデルの再学習を毎回フルで行っており、増分学習（incremental learning、直近データのみで軽量に更新する手法）ではない。XGBoost/LightGBMには`xgb_model=`引数などでウォームスタート（既存モデルに追加の木を生やす）機能があり、これを使えば再学習コストを大幅に下げられる可能性がある。

## 一言まとめ

学習ベースのグラフ構築＋厳密なマッチングアルゴリズム（Biohub）、多様なモデルファミリーを束ねる貪欲アンサンブル（Playground）、そして「時系列のオンライン学習では検証データの取り方の常識が通用しない」という実務的な教訓を含む金融時系列（Hull Tactical）という3本を並べることで、グラフ最適化・アンサンブル手法・時系列特有の検証設計という異なる切り口の技術を学べる組み合わせになりました。3本とも、Codeタブ上位に評価指標の境界を突いた「メトリックハック」的なnotebook群が存在しており、スコアの高さだけでなく手法の中身を確認してから学ぶことの重要性が改めて浮き彫りになりました。
