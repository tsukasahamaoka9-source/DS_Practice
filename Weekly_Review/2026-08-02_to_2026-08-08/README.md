# 週次まとめ 2026-08-02 〜 2026-08-08

## 1. 対象期間・扱ったコンペ一覧

今週は日曜(08-02)〜月曜(08-03)朝まではDaily_Kaggle_Reviewの実績なし。稼働したのは以下の5日間、計15本。

### [2026-08-03](../../Daily_Kaggle_Review/2026-08-03/README.md)
- Biohub: [Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) — Learned Graph w Gap Recovery
- Playground: [Predicting Smartphone Addiction (S6E8)](https://www.kaggle.com/competitions/playground-series-s6e8) — Hill Climbing Ensemble
- 実コンペ: [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) — Ensembling Trees with Online Training

### [2026-08-04](../../Daily_Kaggle_Review/2026-08-04/README.md)
- Biohub: biohub ct mix divaug
- Playground: What actually moved the score on S6E8 — and what didn't
- 実コンペ: Hull Tactical - Market Prediction — EDA which makes sense

### [2026-08-05](../../Daily_Kaggle_Review/2026-08-05/README.md)
- Biohub: Biohub Cell Tracking Solution
- Playground: S6E8 honest OOF blend
- 実コンペ: Hull Tactical - Market Prediction — HTMP-OPTUNA-V8-FINAL-CPU

### [2026-08-06](../../Daily_Kaggle_Review/2026-08-06/README.md)
- Biohub: biohub-v6-ultra-best
- Playground: S6E8 Addiction LB 0.97083
- 実コンペ: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) — DINOsaur V2

### [2026-08-07](../../Daily_Kaggle_Review/2026-08-07/README.md)
- Biohub: Biohub Harmonic Bidirectional Association V1
- Playground: 🥇 #1 Public LB 0.97068 | Honest 55-Model Stack
- 実コンペ: RSNA Knee Abnormality Detection — RSNA-dinov2-ensemble

## 2. 今週繰り返し登場した手法・パターン Top5

1. **メトリックハック（Metric Hacking）への警戒** — Biohubコンペで5日連続（08-03〜08-07）、Codeタブ上位が座標範囲・時刻の妥当性検証漏れを突いた「メトリックハック」notebook群（スコア0.95〜0.97台）で占められている現象が繰り返し確認された。今週選定したnotebookはいずれも意図的に「正攻法スコア」（0.88〜0.92台）のものを選んでおり、スコアの高さより手法の中身を優先する姿勢が一貫していた。
2. **Playground S6E8のOOFブレンド／スタッキングの日々の進化** — 08-03のHill Climbing（6モデル）から始まり、08-04のtarget encoding+ノイズフロア分析、08-05の74モデルhonest OOF blend、08-06の「提出専用」ブレンドnotebook、08-07の55モデルhonest stackと、同一コンペで手法が日を追うごとに大規模化・洗練されていく過程を観察できた。共通して「fold整合性（Honest OOF）」と「ノイズフロアを超える改善か」の2点を検証する姿勢が軸になっている。
3. **検出→ILP/Hungarian法によるグラフ最適化（Biohub）** — 3D U-Net（`TemporalUNet3D`）による検出＋Transformerによるエッジスコアリング＋ILP/Hungarian法による後処理という2段階パイプラインが5日連続で基本構造として登場。08-05・08-06ではTTA（8方向flip/rotation）や複数チェックポイントアンサンブルが上乗せされ、08-07では前向き・後向きスコアを調和平均で融合する双方向設計が加わるなど、同じ骨格の上で工夫が積み重なっていった。
4. **ノイズフロア測定によるスコア差の有意性検証** — 08-04（反復CVでの測定誤差の定量化）、08-05（ブートストラップによるLB検出限界の推定）、08-07（55モデル vs 47モデルの僅差から規模とスコアの非比例性を示唆）と3回登場。「小さいスコア差が本当に意味があるか」を先に測ってから施策を採否判断する考え方が今週のPlaygroundレビュー全体を貫いていた。
5. **DINOv2の凍結バックボーンとしての医療画像応用（RSNA Knee）** — 08-06に新規追加された実コンペ枠として初登場し、08-07も連続採用。凍結DINOv2＋診断項目ごとの専用Attentionヘッド（SlotHead）という設計が2日連続で確認でき、正解ラベルが数十件という極小データ設定への基盤モデル転用という新しいテーマが今週後半で立ち上がった。

## 3. 今週の珍しい・特徴的な手法

- **双方向調和平均マッチング（Bidirectional Harmonic-Mean Association Fusion）**（08-07 Biohub）: フレーム間リンクを前向き・後向き独立にスコアリングし、算術平均ではなく調和平均で融合することで、片方向だけ高スコアな「怪しい」候補を自然に減点する設計。今週はこの1本にしか登場しなかった。
- **ルールベース×テキスト分類器のハイブリッド擬似ラベリング**（08-06, 08-07 RSNA Knee）: 7言語対応の正規表現ルールとTF-IDF＋ロジスティック回帰を確信度で組み合わせ、放射線レポートから擬似ラベルを生成する手法。正解ラベルが58件と極端に少ない医療画像コンペならではの工夫だった。

## 4. 今週のGLOSSARY新規追加語

詳細は[GLOSSARY.md](../../GLOSSARY.md)参照。今週は約28語と大幅な追加があった。

**2026-08-03追加:** ROC-AUC, ILP（整数線形計画法）, クロスアテンション, Hill Climbing, オンライン学習・逐次学習（時系列での注意点）

**2026-08-04追加:** ターゲット・頻度エンコーディング（ネストCV）, ノイズフロアの測定, ロジットスケールでのスタッキング, トップハット変換, ウォークフォワード検証, Zarr形式

**2026-08-05追加:** Test-Time Augmentation, Honest OOF, 汎化ギャップ最小化によるハイパーパラメータ選定, 非同質性グラフと調整済み同質性, エッジ中心アテンション（Paper_Digest由来）, オフラインパッケージインストール, TabM

**2026-08-06追加:** マクロ平均AUC, CTCスタイルのトラッキンググラフ評価, 時空間サイン波位置エンコーディング, ギャップクロージング＋短小トラック除去, DINOv2, ルールベース×テキスト分類器のハイブリッド擬似ラベリング, OOFライブラリの持ち込みブレンド, アンサンブルのアンサンブル

**2026-08-07追加:** 双方向調和平均マッチング, 欠損パターン別メタスタッキング, ターゲット別クエリAttentionヘッド（SlotHead）, ランクベースアンサンブル, Config-Driftガード

## 5. 今週の論文消化との接続

今週は[Paper_Digest](../../Paper_Digest)を2件消化した。

**[2026-08-05: Higher-Order Cell Tracking Transformer](../../Paper_Digest/2026-08-05/README.md)**（Biohub San Francisco著者陣、Kaggleコンペ主催組織そのもの）。候補グラフの調整済み同質性が`H_adj ≈ 0.01`（ランダムと区別不可能）であることを実測し、GNNベースの検出・リンク一体化学習が原理的に機能しにくいことを定量的に裏付けた。直近のBiohubレビュー（08-03など）で「将来の発展方向」として挙げていたGNN系アプローチの優先度を下げる強い根拠になった一方、ILPソルバーの2つの改良（可変出現確率・2段階トラックレット解法）は既存パイプラインへの移植コストが低く、08-06・08-07で見た`GAP_GATE_UM`等のハードコード閾値の改善余地に直接応用できそうだと分かった。

**[2026-08-08: OrthoDiffusion](../../Paper_Digest/2026-08-08/README.md)**（筋骨格系MRI向け拡散モデル基盤モデル）。08-06・08-07に扱った「RSNA Knee Abnormality Detection」と同一ドメイン（膝MRI・極小ラベル設定）で、DINOv2系の対照学習とは異なる「拡散モデルのノイズ除去過程の中間表現」を凍結特徴量として使う戦略を提示。ラベル10%でも性能劣化が小さいという定量結果があり、既存notebookのDINOv2特徴に第二の特徴抽出器として拡散モデル系の中間活性化を加えるアンサンブル多様化の着想として参考になる。

## 6. 一言まとめ

今週最大の特徴は「スコアの裏側を疑う」というテーマがBiohub・Playground・Hull Tactical・RSNA Kneeの全コンペを横断して繰り返し現れたこと。Biohubのメトリックハック、Playgroundのノイズフロア測定とHonest OOF、Hull Tacticalの「リーダーボードは完全にミスリーディング」という著者自身の警告、RSNA Kneeの「複雑な手法より先に擬似ラベルの質を疑う」姿勢——切り口は違えど根は同じだった。また、08-05のPaper_Digestが自コンペ主催者による論文でGNN路線を定量的に否定する根拠を与えてくれたのは、実践と理論がうまく噛み合った収穫。来週は、Biohubで移植候補として挙がったILPの2改良（可変出現確率・2段階トラックレット解法）を実際に手元で試す形の振り返りができると理想的。RSNA Kneeも2日連続で採用されたため、来週以降も定点観測を続けたい。
