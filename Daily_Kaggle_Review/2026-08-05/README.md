# 2026-08-05 Kaggle日次レビュー

固定枠のBiohub 1本、Playground Series 1本、開催中の実コンペ1本、計3本の高スコアnotebookを解説付きでipynb化しました。

## 1. Biohub - Cell Tracking During Development

- **notebook**: [Biohub Cell Tracking Solution](https://www.kaggle.com/code/kaiwalyaatulraut/biohub-cell-tracking-solution) by KAIWALYA RAUT
- **スコア**: notebook詳細ページ表示で Public Score 0.885 / Best Score 0.901（V4時点）。コンペCodeタブの一覧（Best Scoreソート）では同notebookが0.966と表示されており、バージョン間の差異と思われる（本レビューでは実行しておらず断定不可）。

**学べる主要テクニック**
- 3D U-Net（`TemporalUNet3D`）＋Transformer（`SimpleNodeTransformer`）によるノード検出＋エッジ予測の2段構成
- sin/cos位置エンコーディングで時空間座標をTransformerに渡す工夫
- 8方向flip/rotationのTest-Time Augmentation（TTA）で検出を安定化
- ILP（整数計画法）ソルバーで大域最適なリンクグラフを構築（貪欲法より高精度）
- ハンガリアン法による1フレームギャップのクロージング処理と短小トラックの除去

**評価指標の要約**: ノード検出（node_recall）とエッジ・分裂イベントのTP/FP/FN（edge_jaccard等）を組み合わせたCTC系のグラフ評価指標。検出精度と時系列リンクの整合性を同時に評価する設計。

**改善点の考察**
- 同コンペのCodeタブ上位には`biohub-dual-seed-adapted-v1`や`Biohub Dual Seed Frame Retention Guard V1`など「複数シードでのアンサンブル」を行うnotebookがあり、本notebookは単一シード・単一モデルのみで、シードアンサンブルによる分散低減の余地がある。
- 上位には`biohub-v6-ultra-best`のような反復改善版も見られ、本notebookはハイパーパラメータ探索（`POINT_THRESHOLD`や各種距離閾値）がハードコードされたままで、体系的なチューニングの形跡が薄い。
- 直近の研究（[GNN for Cell Tracking, arXiv:2202.04731](https://arxiv.org/abs/2202.04731)、[Segment Anything for Cell Tracking, arXiv:2509.09943](https://arxiv.org/pdf/2509.09943)、TRACKASTRAなど）では、Graph Neural Networkでノード・エッジ特徴を同時にメッセージパッシングで更新する手法や、Segment Anythingベースの汎用セグメンテーションを追跡に転用する手法が提案されており、本notebookのTransformerベースの2段推論より一体化した設計への発展余地がある。
- 最後の`augment_dataset`セルは、実際の追跡とは無関係な負の時刻・範囲外座標のダミーノード/エッジを追加しており、採点スクリプトの特性を突いた可能性が高い（コンペDiscussionにも"Metric hack"系notebookが複数投稿されている）。学習用にはむしろ「なぜスコアが変わるのか」を検証する材料として有用。
- CV（手元検証）とLB（公開リーダーボード）のギャップ検証コードは用意されているが`MODE=="local"`のときしか動かず、実際にどの程度ギャップがあるか本notebook単体では確認できない。

## 2. Playground Series S6E8: Predicting Smartphone Addiction

- **notebook**: [S6E8 honest OOF blend](https://www.kaggle.com/code/szymonkapiski/s6e8-honest-oof-blend) by szymonkapiski
- **スコア**: OOF ROC AUC 0.969687 / Public LB 0.97084（Bronze medal, 25 votes）

**学べる主要テクニック**
- 74モデル分のOut-of-Fold予測を集めた「OOFライブラリ」構築とlogit空間でのロジスティック回帰ブレンド
- 全メンバーで`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`を統一し、fold不一致によるリークを徹底排除する「honest CV」設計
- 相関係数に基づくモデル多様性の評価（スコアの高さより相関の低さを重視）
- 欠損数に応じてメタモデルの重みを変える「missingness regime」設計
- ブートストラップによるリーダーボードの検出限界（ノイズフロア）の定量化

**評価指標の要約**: ROC AUC。閾値に依存せず予測順序の正しさを評価する、二値分類の標準指標。

**改善点の考察**
- 同コンペには`S6E8 full lattice target encoding with XGBoost`（0.96926）のような、格子状target encodingに特化した単一モデルnotebookがあり、本notebーク内の`lat_xgb`系モデルと似た発想だが、より体系的に格子解像度を変えた比較が可能かもしれない。
- `S6E8: HistGradientBoosting | LB 0.96914`のような単一モデルの高スコアnotebookもあり、74モデルという規模の割に「単一モデルとの差」がどこまで統計的に有意か（本notebook自身が指摘するノイズフロア0.00014との比較）を明示するとより説得力が増す。
- 直近の研究（[TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling, ICLR 2025](https://github.com/yandex-research/tabm)）は、本notebookが使っている`TabM`系モデルの元論文であり、単一ネットワークでアンサンブル相当の効果を効率的に得る手法。ブレンドに使う個々のモデルの改善余地として、この論文の設計指針（重み共有型アンサンブル）をさらに追い込む余地がある。
- Leave-One-Out診断（`RUN_LOO`）は約1時間かかるため事前計算結果を貼り付けているが、ライブラリのメンバーが変わるたびに再計算が必要になり、再現性の維持コストが高い。軽量な近似指標（相関行列だけからの推定など）で代替できないか検討の余地がある。
- `lookup`モデル1つの寄与がブレンド全体の0.000106と圧倒的に大きく、単一障害点（single point of failure）になっている。このモデルの学習が失敗した場合のフォールバック戦略が明記されていない。

## 3. Hull Tactical - Market Prediction

- **notebook**: [HTMP-OPTUNA-V8-FINAL-CPU](https://www.kaggle.com/code/frankmorales/htmp-optuna-v8-final-cpu) by frankmorales
- **スコア**: ローカル検証 1.06937 / Kaggle Score 1.053（Silver medal, 51 votes）

**学べる主要テクニック**
- Polarsによる大規模時系列特徴量エンジニアリング（ラグ・移動窓統計・日内ランク/Zスコア・交互作用、計1187列）
- XGBoost・LightGBM・CatBoostの3モデルアンサンブル（重み付き平均、Optunaで重みも同時探索）
- 予測確率を「0.5からの距離＝自信度」として配分量（0〜2倍のレバレッジ）に変換する設計
- Optuna 3000試行超のハイパーパラメータ探索で「検証スコアの高さ」より「本番との汎化ギャップの小ささ」を重視したモデル選定
- `kaggle_evaluation`推論サーバーでのエラー時フェイルセーフ（例外時は無配分`0.0`を返す）

**評価指標の要約**: 市場に対する超過リターンのSharpe比を、ボラティリティ超過分と市場未達分の2つのペナルティで調整した指標。リスクを取りすぎず、かつ市場に負けない戦略を評価する設計。

**改善点の考察**
- 同コンペには`Hull Weighted Ensemble No Internet`のような「オフラインで完結する重み付きアンサンブル」notebookや、`Hull Starter Notebook`（公式配布、1740 votes）のようなシンプルなベースラインがあり、本notebookの1187特徴量は前者と比べてかなり大規模。特徴量数と汎化ギャップの関係を系統的に検証した比較表があるとより説得力が増す。
- `YOUR NOTEBOOK vs THE 17.507 BEAST🔥`のようなタイトルのnotebookが投稿されており、コミュニティ内で極端に高いスコア（おそらく指標の特性を突いたもの）への警戒が話題になっている。本notebookのスコア1.053は堅実だが、上位との差がどのような手法差から来ているかの分析は含まれていない。
- 直近の研究（[Causal and Predictive Modeling of Short-Horizon Market Risk, arXiv:2510.22348](https://arxiv.org/pdf/2510.22348)ではニューラルネットとツリー系モデルを組み合わせたハイブリッドアンサンブルでSharpe比2.51を達成、[A machine learning approach to risk based asset allocation, Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-26337-x)ではLSTMによるボラティリティ予測とリスク予算配分層を組み合わせて伝統的リスクパリティを55%上回るSharpe比を報告しており、本notebookの「確信度→配分量」という単純な線形変換より、動的リスク予算配分のような手法に発展させる余地がある。
- LightGBM・CatBoostがEarly Stoppingなしで固定5000イテレーションまで学習しており、XGBoostだけ検証ベースのEarly Stoppingを使うという非対称設計の妥当性が検証されていない（他2モデルも同様に検証すれば安定性が増す可能性）。
- Optunaの探索履歴（Trial 397, 957, 2239, 3104, 3258, 3319, 3455）から見ると汎化ギャップが正の値（LBが検証を上回る）になるケースもあり、本当に「安定した設定」なのか、単に運が良かっただけなのかを区別する統計的な検定（例えば複数シードでの再学習によるギャップの分散推定）が追加できるとより信頼性が増す。

---

## 今日の一言まとめ

3本を通じて共通していたのは「スコアの数字だけでなく、その数字がどう作られているかを疑う姿勢」でした。BiohubのダミーノードのようなグレーなテクニックからPlaygroundの「honest CV」という誠実な検証、Hull Tacticalの汎化ギャップ重視のモデル選定まで、高スコアの裏にある設計思想の違いが今日いちばんの学びです。
