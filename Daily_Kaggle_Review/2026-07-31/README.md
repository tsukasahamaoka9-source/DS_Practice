# 2026-07-31 Kaggle Daily Review

今日は固定枠のBiohub、Playground Series、開催中の実コンペ(金融)の3本を解説付きノートブック化しました。

## 1. Biohub: Multi-Scale Blob Tracking With Robust Centroids

- notebook: `biohub_biohub-cell-tracking-data-model-eda-baseline.ipynb`
- 元notebook: [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline) by Pilkwang Kim(144 upvotes, Gold, Public/Best Score 0.827)
- コンペ: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)(ゼブラフィッシュ胚発生の3D顕微鏡動画から細胞を検出・追跡、賞金$60,000、開催中の研究コードコンペ)

学べる主要テクニック:
- Difference of Gaussians(DoG)による多重スケールのblob(細胞候補)検出
- 局所背景を差し引いてから重心を再計算する頑健な重心精密化(refine_centroids)
- 物理距離(µm単位)に基づくHungarian法(`linear_sum_assignment`)でのフレーム間リンク
- 1フレームぶんの検出漏れを線形補間で埋めるgap closing
- どのエッジにもつながらない孤立ノードの除去による後処理

評価指標の要約: score = adjusted_edge_jaccard + 0.1×division_jaccard。ノード同士の対応(エッジ)精度をJaccard係数で測り、細胞分裂の検出精度(division Jaccard)を0.1倍の重みで加える設計。分裂イベントは希少なので小さい重みだが無視はできない構成。今回のnotebookはルールベースの検出→リンクのみで、分裂検出(`allow_divisions`)は既定でオフのままだった。

改善点の考察:
- 分裂検出を明示的に有効化していない(`Config.allow_divisions=False`のまま)ため、division Jaccard分のスコアを全く取れていない可能性が高い。まずここを有効化するのが最も費用対効果の高い改善。
- DoGの多重スケールパラメータ(`dog_small_um`, `dog_large_um`)は手動設定値のままで、データセットごとの細胞サイズ分布に応じた自動調整の余地がある。
- Hungarian法のゲート距離(`max_link_um`)や再帰的gap closingの閾値も固定値で、細胞密度に応じた適応的なゲート幅にできる可能性がある(高スコアのBiohub notebook群には密度適応ゲートを実装しているものもあった)。
- 学習済みモデル(UNet等)を使わないルールベース手法のため、高密度領域での誤検出・取り違えには弱い可能性がある。

## 2. Playground: Student Health Risk — LightGBM + Optuna

- notebook: `playground_student-health-risk-lightgbm-optuna-0-95014.ipynb`
- 元notebook: [Student Health Risk — LightGBM + Optuna | 0.95014](https://www.kaggle.com/code/rugvedbane/student-health-risk-lightgbm-optuna-0-95014) by Rugved Bane(38 upvotes, Public/Best Score 0.95014)
- コンペ: [Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)(Playground Series - Season 6 Episode 7。学生の生活習慣データから健康状態を3クラス分類)

学べる主要テクニック:
- `stress_level × physical_activity_level`の交互作用特徴量(LightGBMの重要度で最重要な特徴になった)
- `compute_sample_weight(class_weight="balanced")`によるクラス不均衡対策(Optunaの交差検証内・最終学習の両方で一貫して適用)
- Optuna(ベイズ最適化)による3-fold StratifiedKFoldでのハイパーパラメータ探索。目的関数を評価指標そのもの(balanced accuracy)に直接設定
- リーク防止を意識した前処理(`SimpleImputer`はtrainのみでfit、`LabelEncoder`はtrain+testを結合してfitし未知カテゴリを回避)

評価指標の要約: Balanced Accuracy(各クラスのrecallの平均)。3クラス(at-risk/unhealthy/fit)の分布が偏っている可能性が高く、単純accuracyだと多数派クラスに引っ張られるため採用されていると考えられる。このnotebookは`compute_sample_weight`で明示的にクラス不均衡へ対応している点が評価指標との整合性が高い。

改善点の考察:
- 著者はLightGBM > XGBoost > CatBoostという比較結果を主張しているが、実際の比較コード(XGBoost/CatBoostの学習)はnotebook内に再現されておらず、根拠が示されていない。
- 単一モデル(LightGBM)のみで、他の上位notebook(HGBC/CatBoost/XGBoost/LGBMのスタッキング等)と比べるとアンサンブルの余地が残る。
- SMOTE等のオーバーサンプリングや、予測確率の閾値調整(post-hoc calibration)によるさらなるbalanced accuracy改善の余地がある。
- 交差検証は3-foldとやや少なめで、5-fold程度に増やすことで評価の安定性が上がる可能性がある。

## 3. 実コンペ: Hull Tactical - Market Prediction

- notebook: `competition_deep-eda-smart-feature-selection-ml-models.ipynb`
- 元notebook: [Deep EDA + Smart Feature Selection + ML models](https://www.kaggle.com/code/tungdang1108/deep-eda-smart-feature-selection-ml-models) by TungBayes(105 upvotes, Silver, Best Score 1.480)
- コンペ: [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)(匿名化された市場特徴量からS&P500の翌日超過リターンを予測し0〜2倍のポジションサイズを決定、賞金$100,000、Featured Code Competition、Late Submission受付中)

学べる主要テクニック:
- Granger因果性・階層クラスタリング・レジーム別分析・ローリングウィンドウ・交互作用という5手法のアンサンブルによる特徴量選択(単一の重要度指標に頼らない)
- ハイパーパラメータ探索(Optuna/TPE)とポジションサイズの最適化(`optimize_position_multiplier`、L-BFGS-B)の両方を、代理損失(MSE等)ではなくコンペ本来の評価指標(ScoreMetric)に対して直接最適化している
- ローリング相関・ローリング相互情報量・構造変化検出(structural break detection)による特徴量とターゲットの時変関係の分析
- 時系列データへのリーク防止を意識した、シャッフルなしの時系列順70/20/10分割(`.iloc`スライス)

評価指標の要約: 市場平均に対する幾何平均Sharpe比を、ボラティリティ超過(市場の1.2倍超)へのペナルティとリターン未達への二次ペナルティで割った独自指標。単なるリターンでなくリスク調整後のパフォーマンスを問う設計。なお、このコンペのCodeタブ上位には同一スコア(17.396等)に集中したnotebookが多数あり、評価指標の境界条件を突いた「degenerate戦略」の可能性が高い。今回選んだnotebック(1.480)はそのクラスタと異なる値で、独自の特徴量分析に基づくアプローチと考えられる。

改善点の考察:
- **重大な指摘**: 実際に提出用`predict()`関数で使われている式は`pred_ml*0.10 + pred_signal*0.90`であり、コード中のコメント「PURE ML WITH OPTIMIZED MULTIPLIER」と矛盾する。丹念にチューニングされたMLモデルの寄与は推論時にわずか10%しかなく、大部分はシグナルベースの別ロジックに依存している。検証時のスコアも、既知の`date_id`に対する99%重みのオラクル参照によってかさ上げされている可能性がある。
- train/testの`date_id`重複を示す警告がnotebook自身の出力に現れているが、対応が明示されていない。
- ボラティリティ制約(120%)に対する明示的なポジションサイズの安全マージンの設計が薄く、制約違反ペナルティのシミュレーションがあるとより頑健になる。
- 5手法アンサンブルによる特徴量選択は独自性が高い一方、各手法の重み付け根拠(なぜ単純な多数決/和集合なのか)の妥当性検証(消去法によるアブレーション等)が示されていない。

## 一言まとめ

ルールベースの物理法則に基づく検出・追跡(Biohub)、評価指標を直接最適化するタブular分類の定石(Playground)、そして「見た目のスコアと実装の中身が食い違う」教訓を含む金融時系列の特徴量分析(Hull Tactical)という3本を並べることで、シンプルな手法の伸びしろの見つけ方と、高スコアなノートブックほど中身を疑って読む重要性の両方を学べる組み合わせになりました。
