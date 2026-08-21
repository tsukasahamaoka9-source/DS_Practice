# 週次まとめ 2026-07-26 〜 2026-08-01

## 1. 対象期間・扱ったコンペ一覧

今週は日曜(07-26)〜火曜(07-28)はDaily_Kaggle_Reviewの実績なし。稼働したのは以下の3日間、計8本。

### [2026-07-29](../../Daily_Kaggle_Review/2026-07-29/README.md)
- Playground: [Predicting Student Health Risk (S6E7)](https://www.kaggle.com/competitions/playground-series-s6e7) — Stacked HGBC/CatB/XGB/LGBM
- 実コンペ: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) — Nearest Neighborベースライン

### [2026-07-30](../../Daily_Kaggle_Review/2026-07-30/README.md)
- Playground: [Predicting Stellar Class (S6E6)](https://www.kaggle.com/competitions/playground-series-s6e6) — Galaxy Population XGB
- 実コンペ: [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) — Hull Starter Notebook

### [2026-07-31](../../Daily_Kaggle_Review/2026-07-31/README.md)
- Biohub: Multi-Scale Blob Tracking With Robust Centroids
- Playground: Student Health Risk — LightGBM + Optuna (0.95014)
- 実コンペ: Hull Tactical - Market Prediction — Deep EDA + Smart Feature Selection

## 2. 今週繰り返し登場した手法・パターン Top5

1. **勾配ブースティング系モデル（XGBoost / CatBoost / LightGBM）** — Playground3日連続で主力モデルとして登場（07-29: XGB+CatBoostアンサンブル、07-30: XGBClassifier単体、07-31: LightGBM+Optuna）。表形式データの分類ベースラインとして依然デフォルトの選択肢であることが確認できた。
2. **Hungarian法（`scipy.optimize.linear_sum_assignment`）によるフレーム間対応付け** — Biohubの2本（07-29のNearest Neighborベースライン、07-31のDoG多重スケール版）どちらのコアにも使われており、シンプルなベースラインから精度を上げても「検出→リンク」の骨格は変わらず、検出側（DoG、gap closing等）の作り込みでスコアが伸びることが分かった。
3. **クラス不均衡対策（balanced accuracy / `class_weight="balanced"` / `compute_sample_weight`）** — Student Health Riskコンペで07-29・07-31の2回とも中心的な論点。評価指標自体がbalanced accuracyであるため、指標と前処理の整合性を取ることが素点に直結すると再確認できた。
4. **リーク防止を意識したCV・前処理設計** — 07-29のfold単位ターゲット統計、07-31 Playgroundの「`SimpleImputer`はtrainのみfit・`LabelEncoder`はtrain+test結合」、07-31 Hull Tacticalの時系列シャッフルなし70/20/10分割、と3件で繰り返し登場。データの性質（表形式 or 時系列）によってリークの形が変わる点が学びになった。
5. **Optuna（ベイズ最適化）による直接最適化** — 07-31に2本連続で登場（Playgroundのハイパーパラメータ探索、Hull Tacticalのポジションサイズ最適化）。いずれも代理損失ではなく評価指標そのものを目的関数にしている点が共通しており、今週新規にGLOSSARY入りした概念でもある。

## 3. 今週の珍しい・特徴的な手法

- **Granger因果性検定・構造変化検出（Structural Break Detection）による時系列特徴量選択**（07-31 Hull Tactical）: 単純な相関ではなく「先行して動くか」「レジームが変わっていないか」を統計的に検定してから特徴量を絞り込むアプローチ。金融時系列特有の高度な手法で、今週はこの1本にしか登場しなかった。
- **「スコアと実装内容の食い違いを疑って読む」教訓**（07-31 Hull Tactical）: 提出用`predict()`が実コード上`pred_ml*0.10 + pred_signal*0.90`となっており、notebook内コメント「PURE ML WITH OPTIMIZED MULTIPLIER」と矛盾していた点。Sharpe比の項でも触れた通り、多数のnotebookが同一スコアに集中する「メトリックハック」の兆候と合わせ、高評価のnotebookほど中身を検証する重要性を再認識させられた回だった。

## 4. 今週のGLOSSARY新規追加語

詳細は[GLOSSARY.md](../../GLOSSARY.md)参照。

**2026-07-31追加（Daily_Kaggle_Review由来）:**
- 評価指標: Jaccard係数, Balanced Accuracy, Sharpe比
- 手法: Difference of Gaussians（DoG）, Hungarian法, クラス重み付け（`compute_sample_weight`）, Granger因果性検定, 構造変化検出（Structural Break Detection）
- ライブラリ・ツール: Optuna

**2026-08-01追加（Paper_Digest由来）:**
- 評価指標: CVaR（Conditional Value at Risk）, Omega比
- 手法: 微分可能サロゲート目的関数（Differentiable Surrogate Objective, softplus近似）

## 5. 今週の論文消化との接続

2026-08-01に[Paper_Digest](../../Paper_Digest/2026-08-01/README.md)として「Financially Guided Deep Portfolio Optimization」（Fernandes & Desell, arXiv 2026-05）を1件消化した。ニューラルネットがポートフォリオ配分比率を直接出力し、Sharpe比・Omega比をsoftplus近似で微分可能にした上でCVaR正則化・リスクパリティ正則化と組み合わせて直接最適化するend-to-endフレームワークを提案する論文。

Kaggleとの接続: 07-31にレビューしたHull Tactical - Market Predictionの上位notebookは、既に「代理損失（MSE）でなく評価指標そのもの（Sharpe型のScoreMetric）を直接最適化する」アプローチを採っていたが、本論文はその発想を理論的に一段掘り下げたものと言える。特にsoftplus近似によるSharpe比/Omega比の微分可能化は、Hull Tacticalのような独自評価指標を持つコンペでGBDT+Optunaの目的関数をより滑らかな代理損失に置き換える際にそのまま応用できそうだ。また、CVaR正則化・リスクパリティ正則化のような「主目的だけを追わせない複合損失」の設計思想は、Hull Tacticalで見られた「メトリックハック（同一スコアに集中するdegenerate戦略）」への対策としても参考になる。

## 6. 一言まとめ

今週は「同じコンペを2〜3日にわたって深掘りする」パターン（Biohub・Hull Tacticalとも複数日登場）が特徴的で、ベースラインから徐々に手法が高度化していく過程を見られたのが収穫。土曜（08-01）にはPaper_Digestで1件、Hull Tacticalの実践と直結する論文を消化でき、実践と理論がうまく繋がった週にもなった。来週はこの「Kaggleで見た手法を論文で裏取りする」流れを続けつつ、Paper_Digestの頻度自体をもう少し増やしたい。また、Hull Tacticalで得た「高スコアnotebookほど実装を疑う」姿勢は他コンペのnotebook選定でも意識していきたい。
