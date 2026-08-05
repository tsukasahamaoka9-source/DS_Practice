# 2026-07-30 Kaggle Daily Review

今日はPlayground Series 1本と、実コンペ（金融・市場予測）1本を解説付きノートブック化しました。

## 1. Playground: Predicting Stellar Class (Season 6, Episode 6)

- notebook: `playground_galaxy-population-xgb.ipynb`
- 元notebook: [Galaxy Population XGB](https://www.kaggle.com/code/mpwolke/galaxy-population-xgb) by Marília Prata（55 upvotes, Silver）
- コンペ: [Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6)（SDSS17のスペクトルデータから天体をGALAXY/QSO/STARの3クラスに分類）

学べる主要テクニック:
- 天文学ドメイン知識（赤方偏移、恒星のスペクトル型、銀河のレッドシーケンス/ブルークラウド/グリーンバレー）を踏まえたEDA
- カテゴリ変数・目的変数の分布可視化によるクラス不均衡の把握
- 数値特徴量の相関ヒートマップによる多重共線性の確認
- 文字列ラベルの数値エンコーディングと不要列の除外
- `XGBClassifier` + early stoppingによるシンプルな分類ベースライン（Accuracy ≈ 0.968）

## 2. 実コンペ: Hull Tactical - Market Prediction

- notebook: `competition_hull-starter-notebook.ipynb`
- 元notebook: [Hull Starter Notebook](https://www.kaggle.com/code/laurentlanteigne/hull-starter-notebook) by Laurent Lanteigne（1738 upvotes, Gold、コンペ公式ピン留めスターター）
- コンペ: [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)（匿名化された市場特徴量から翌日の市場超過リターンを予測し取引シグナルに変換、賞金$100,000、Featured Code Competition）

学べる主要テクニック:
- `dataclass`による設定値・データの型安全な一元管理（バリデーション付き・frozenによる不変化）
- `polars`を使った高速な特徴量エンジニアリング（差分・比率特徴量、指数加重移動平均での欠損補完）
- `ElasticNetCV`による正則化強度の自動探索とその結果を使った本番モデルの学習
- 予測リターンを`np.clip`で安全な範囲の取引シグナルに変換するリスク管理の考え方
- `kaggle_evaluation`を使ったCode Competition向け推論サーバーの実装パターン

## 一言まとめ

天文学の分類問題と金融市場の回帰問題という全く異なる分野を並べることで、「ドメイン知識を踏まえた素直なEDA→定番モデルでのベースライン構築」という基本の型と、「設定管理・前処理の共通化・提出用インターフェース実装」という実務寄りのパイプライン設計、両方の観点を1日で学べる組み合わせになりました。
