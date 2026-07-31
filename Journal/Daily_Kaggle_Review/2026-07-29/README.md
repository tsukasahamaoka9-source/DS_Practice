# 2026-07-29 Kaggle Daily Review

今日はPlayground Series 1本と、実コンペ（研究コード / 生物学）1本を解説付きノートブック化しました。

## 1. Playground: Predicting Student Health Risk (Season 6, Episode 7)

- notebook: `playground_health-stacked-hgbc-catb-xgb-lgbm-baseline.ipynb`
- 元notebook: [Health | Stacked HGBC/CatB/XGB/LGBM | Baseline](https://www.kaggle.com/code/kospintr/health-stacked-hgbc-catb-xgb-lgbm-baseline) by Ákos Pintér（102 upvotes）
- コンペ: [Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)（学生の生活習慣データから健康状態を3クラス分類）

学べる主要テクニック:
- `ColumnTransformer` + `Pipeline`によるスケーリング／エンコーディング／統計特徴量の一元管理
- リーク防止を意識したfold単位のターゲット統計量（mean/std/skew/count等）の算出
- 層化K-Fold（`StratifiedKFold`）によるクロスバリデーション設計
- XGBoost・CatBoostなど複数モデルをOut-of-Fold予測ベースで重み付きアンサンブル
- balanced accuracyでのサブカテゴリ別・複合カテゴリ別のモデル弱点分析

## 2. 実コンペ: Biohub - Cell Tracking During Development

- notebook: `competition_cell-tracking-getting-started-w-nearest-neighbor.ipynb`
- 元notebook: [Cell Tracking Getting Started w/ Nearest Neighbor](https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor) by inversion（Kaggle公式、349 upvotes、Pinned）
- コンペ: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)（ゼブラフィッシュ胚発生の3D顕微鏡動画から細胞を検出・追跡、賞金$60,000、開催中）

学べる主要テクニック:
- Zarr形式で圧縮された大容量3D+時間の顕微鏡データの読み込み（blosc2解凍）
- ダウンサンプリング＋平滑化＋パーセンタイル閾値による簡易2値化セグメンテーション
- `scipy.ndimage.label`による連結成分ラベリングで細胞候補（重心座標）を検出
- フレーム間の細胞対応付けをハンガリアン法（`scipy.optimize.linear_sum_assignment`）で解く「検出→追跡」の基本パイプライン
- 物理スケール（マイクロメートル単位）を考慮した距離計算

## 一言まとめ

同じ「表形式データの分類」でも作り込み度合いが全く異なる2本（洗練されたアンサンブルパイプライン vs. シンプルな検出・追跡ベースライン）を並べることで、前処理・検証設計の丁寧さと、タスクに応じた最小限で筋の良いアプローチの両方の勘所を学べる組み合わせになりました。
