# 2026-08-10 Kaggle日次レビュー

固定コンペ「Biohub - Cell Tracking During Development」1本 + Playground Series 1本 + 開催中の実コンペ1本、計3本を解説付きipynb化した。

## 1. Biohub - Cell Tracking During Development

- **notebook**: [No-Hack | Biohub Cell Another Approch 3rd](https://www.kaggle.com/code/yusuketogashi/no-hack-biohub-cell-another-approch-3rd) by Yusuke Togashi（Pilkwang Kim氏の一連の公開notebookをCopy & Edit）
- **スコア**: Public/Best Score 0.915（クリーンなリーダーボード上）
- **主要テクニック**:
  - 既存のクリーンな0.915ベースライン（「Biohub 159B」）を完全に凍結し、新規要素を1つだけ足すという厳密な実験設計
  - 22特徴量の学習済み「ローカル関連性ランカー」による候補リンクの再スコアリング
  - 8方向（回転・転置込み）のTest-Time Augmentationによる検出の安定化
  - 整数線形計画法（ILP）によるグラフ最適化＋ギャップ補完・分裂ジオメトリチェックなどの後処理
  - 新規要素: 3フレーム先読みの加速度整合性ボーナス（候補リンクの先で滑らかに継続する軌道を優遇）
- **評価指標の要約**: CTC系のnode_recall（検出率）とedge_jaccard（リンクの一致度）に相当する指標で、検出とトラッキングの精度を分離して評価する。本notebookは検出・グラフ最適化は既存のまま、「未来方向の運動の滑らかさ」という新しい評価軸をエッジ選択に加えることでedge_jaccardの改善を狙っている。
- **改善点の考察**:
  1. **他notebookとの比較**: 同コンペのCodeタブを見ると、上位（0.95〜0.97台）はタイトルに`Metric_hack`と明記された、座標範囲外・負の時刻に偽ノード/エッジを追加してスコアを不正に稼ぐnotebook群が占めている（GLOSSARY.mdの「メトリックハック」参照）。一方、「クリーンな」到達点はほぼ0.915付近（144チームが同スコアに集中）に固まっており、本notebookもその1つ。同じく0.915台の「Biohub Harmonic Bidirectional Association V1」（過去に選定済み）は前向き・後向き両方向のリンクスコアを調和平均で融合する手法を採るが、本notebookはリンクの方向融合ではなく未来方向の物理的整合性のみを追加する、異なるアプローチを取っている。
  2. **関連文献**: [Graph Neural Network for Cell Tracking in Microscopy Videos (ECCV 2022 / arXiv:2202.04731)](https://arxiv.org/abs/2202.04731) は、細胞インスタンスをノード、フレーム間の対応をエッジとしたグラフ全体にメッセージパッシングを行うGNNベースの手法を提案しており、単一フレーム先読みではなく、GNNのメッセージパッシングによって「連続する複数フレームにまたがる情報」を自然に統合する点で、本notebookの手動設計された1フレーム先読みボーナスよりも一般化された発想と言える。GLOSSARY.mdの「エッジ中心アテンション」で扱った非同質性グラフ向けTransformer型アーキテクチャとも同じ問題意識を共有している。
  3. 気づいた点: 本notebook自身のタイトルにある通り「Three-Frame」を謳いながら実装はt+1フレームまでしか見ておらず、t+2まで含めた真の3フレーム先読みには拡張余地がある。また、ギャップ補完時の中間点精緻化が画像の明るさの重心のみに依存しており、GLOSSARY.mdの「Difference of Gaussians」のような形状ベースの検出情報を併用すればさらに頑健になる可能性がある。
  - **改善提案**:
    - t+2フレームまで見る真の3フレーム先読みへの拡張
    - GNNベースのメッセージパッシング（引用文献）によるエッジスコアリングへの置き換え検討
    - ギャップ補完の中間点精緻化に形状情報（DoGなど）を追加
    - 「144チームが0.915に集中」という現象自体を深掘りし、次の有意な改善（0.916超え）にどのメカニズムが必要かを、Discussionタブの議論も踏まえて特定する
    - 学習済みDeepCenterモデル（コード中に存在するが今回は無効化`BIOHUB_USE_DEEPCENTER_VETO=0`）を有効化した際の効果検証

## 2. Playground Series S6E8: Predicting Smartphone Addiction

- **notebook**: [S6E8: Elite Rank Average Ensemble \[0.97092\]](https://www.kaggle.com/code/amanatar/s6e8-elite-rank-average-ensemble-0-97092) by Aman Atar
- **スコア**: Public/Best Score 0.97092（ROC-AUC）
- **主要テクニック**:
  - 4つの公開「エリート」submissionの相関行列で多様性を確認
  - 生の確率値ではなくパーセンタイルランクに変換してから単純平均する Rank Average アンサンブル
  - 動的なCSVファイル探索によるモデル読み込み
- **評価指標の要約**: ROC-AUC（順位のみが評価される指標）。指標の性質上、確率の絶対値ではなく順位の一致度だけが意味を持つため、Rank Averageという手法選択自体が指標への理解を反映している。
- **改善点の考察**:
  1. **他notebookとの比較**: 本notebookが参照する4つの元submission（Najiama's 0.97086, Dariush's "the strongest fully reproducible stack" 0.9708, Szymon's "honest OOF blend" 0.9708, Boltuzamaki's "Stacking 47 models" 0.97073）を見ると、元の個別モデルは「47モデルスタッキング」のような本格的なメタモデル学習を行っているのに対し、本notebookはその4つの最終出力を単純にランク平均するだけに留まっている。単純平均ではなく、ロジスティック回帰などの軽量なメタモデルで重み付けする（スタッキング）方が理論的にはさらに伸びる余地がある。
  2. **関連文献**: Rank aggregationやアンサンブル多様性に関する研究（例: Hybrid Rank Aggregation, biorxiv 2022）では、単純な順位平均よりも教師あり的にランクを重み付けする手法(Supervised Rank Aggregation)がより高い性能を示すことが報告されている。Kaggleの実践例（[Playground S5E12 1st Place: Hill Climbing + Ridge Ensemble](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl)）でも、単純平均よりHill ClimbingやRidge回帰による重み最適化の方が高スコアを達成している。
  3. 気づいた点: 相関行列で「2つのクラスタ、クラスタ間相関0.88」という多様性を確認しているが、クラスタ内でどのモデルが最も情報量を持つかまでは踏み込んでおらず、単純に4モデル均等重みにしている。
  - **改善提案**:
    - 単純ランク平均から、OOF予測を使ったRidge回帰やHill Climbingによる重み最適化への切り替え
    - 4モデルに限定せず、相関が低い（多様性の高い）5つ目以降のモデルを追加候補として探索
    - GLOSSARY.mdの「Honest OOF」の観点から、4つの元submissionの学習に使われたfold分割が一致しているかの検証（fold不整合によるリークがブレンドスコアを不当に押し上げていないかの確認）

## 3. RSNA Knee Abnormality Detection

- **notebook**: [RSNA Knee: read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) by prvsiyan
- **スコア**: Public/Best Score 0.894（V21時点）
- **主要テクニック**:
  - 9言語対応のルールベース読影レポート抽出器で、58件しかない正解ラベルを4,407件の学習用ターゲットへ拡張（弱教師あり学習）
  - DINOv2バックボーン＋所見別クエリAttentionヘッド（SlotHead）による画像専用モデル
  - DICOMヘッダーからの左右laterality幾何学的復元、断面別の左右正規化
  - モデルの「指紋（fingerprint）」照合によるロード時バグの検知
  - GPU互換性の実行時プローブ、フェイルセーフな提出ファイル生成
- **評価指標の要約**: 12所見それぞれのROC-AUCのマクロ平均。順位のみが評価されるため、複数モデルの統合もパーセンタイルランクで行っている。
- **改善点の考察**:
  1. **他notebookとの比較**: 同コンペで本notebookより高いスコア（0.899）を出している「[0.899] Let me Cook」や「RSNA Knee Baseline V1」は、テキストによる弱教師あり学習ではなく、既存の公開ベースライン（Pilkwang Kim氏の20メンバーアンサンブル、0.891）に対してTTAやプーリング手法の調整（`rsna-knee-baseline-v1-fracture-tta-pool-probe`など）で純粋にinference側を改善するアプローチを取っている。本notebook自身もV12で「Fracture study poolingをmeanからmaxへ変更」しただけで0.891→0.893まで伸びており、テキスト由来ラベルの導入（V14、+0.001）より、実はシンプルなプーリング変更の方が効果が大きかった、という点は興味深い。
  2. **関連文献**: [PromptRad (arXiv:2605.20052)](https://arxiv.org/pdf/2605.20052) は低リソース設定での放射線レポートラベリングにナレッジ強化されたプロンプトチューニングを使う手法を提案しており、本notebookの正規表現ベースの抽出器をLLMベースの抽出（プロンプトチューニング）に置き換えることで、九言語対応の作り込みコストを削減しつつカバレッジを上げられる可能性がある。また、[Weakly Supervised Deep Learning in Radiology (RSNA Radiology誌)](https://pubs.rsna.org/doi/full/10.1148/radiol.232085) は弱教師あり学習の一般的な留意点をまとめており、レポートの「無言及」を「陰性」とみなす仮定の妥当性検証は本notebook自身も認識している通り重要な論点。
  3. 気づいた点: B3ブランチがv42→v43→v46と3回書き直されている経緯は、Kaggle環境のGPU割り当ての不確実性（P100とTorchカーネルの非互換）に強く依存しており、環境要因による手戻りコストの大きさを物語っている。
  - **改善提案**:
    - Fracture以外の所見についても、meanプーリングからmaxプーリングへの変更効果を系統的に検証する（V12の学びの横展開）
    - テキスト抽出器をLLMベースの手法（引用文献のPromptRad等）に置き換え、9言語分の正規表現メンテナンスコストを削減
    - 「無言及＝陰性」という仮定を、所見ごと（特にSynovitisのような言及率が低い所見）に個別検証する
    - GPU互換性チェックの結果を実行開始時に一度だけ判定し、B3ブランチの構成（v42/v43/v46）を動的に1つだけ選択する設計にすることで、リトライの手戻りを減らす

## まとめ

3本に共通するのは、「指標の数学的性質を理解した上で設計を単純化する」姿勢（Biohubの構造的整合性監査、Playgroundのランク統合、RSNAの弱教師あり学習）と、「静かな失敗を防ぐ安全装置」（Config-Driftガード、フェイルセーフ提出、モデルの指紋照合）を随所に組み込む実務的な堅牢性である。一方で、いずれのnotebookも「メトリックハックのような手軽な近道」を避け、地道な改善を積み重ねている点は、学習素材として一貫した価値がある。
