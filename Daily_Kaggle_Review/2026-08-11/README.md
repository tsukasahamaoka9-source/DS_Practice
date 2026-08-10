# 2026-08-11 Kaggle日次レビュー

固定コンペ「Biohub - Cell Tracking During Development」1本 + Playground Series (S6E8) 1本 + 開催中の実コンペ（RSNA Knee Abnormality Detection）1本、計3本の高スコアnotebookを解説付きipynb化しました。

---

## 1. Biohub - Cell Tracking During Development: Cell tracking

- **コンペ**: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- **原著者**: MR. BRUCE ([muhammaddanyalmalik](https://www.kaggle.com/muhammaddanyalmalik))
- **元notebook**: [Cell tracking](https://www.kaggle.com/code/muhammaddanyalmalik/cell-tracking)
- **スコア**: Public/Best Score **0.877**

**学べる主要テクニック**
- オフライン環境向けの依存ライブラリ手動インストール（`pip install --no-index --find-links`）と、`sys.modules`パージによる二重インポート事故の防止
- 検出しきい値・ILP重み・後処理パラメータを`V5〜V9`の5パターンで体系的に比較する「実験管理」的なハイパーパラメータ運用
- 整数線形計画法（ILP）によるフレーム間リンクのグラフ最適化と、8方向のTTA（Test Time Augmentation）
- 提出フォーマットの構造要件（分岐構造）を満たすためのダミーノード/エッジ「パディング」処理

**評価指標の要約**: ノード検出の再現率（node_recall）とエッジ（リンク・分裂）の一致度を組み合わせたCTCスタイルの複合指標。検出しきい値の上げ下げがrecallと誤検出のトレードオフに直結するため、V5〜V9の比較実験はこのトレードオフを数値で追い込む作業と言える。

**改善点の考察**
- 同コンペのBest Score上位（0.96台）のnotebook群は、本notebookが使うU-Net+Transformer+ILPのパイプラインに加えて、「ローカル関連性ランカー」（22特徴量による候補リンク再スコアリング、8/10のGLOSSARY参照）や「3フレーム先読み加速度整合性ボーナス」といった追加の後処理ステージを積んでおり、本notebookはそこまでの積み増しをしていない。V5〜V9の比較はしきい値レベルの調整に留まっており、パイプライン自体への新規モジュール追加という上位陣のアプローチとは方向性が異なる。
- 細胞追跡分野では2025年に "Cell-TRACTR"（Transformerベースのend-to-endセグメンテーション＋追跡モデル、*PLOS Computational Biology*）が発表されており、検出とリンク予測を別々のステージで行う本notebookの構成に対し、両者を単一のend-to-endモデルに統合する方向性が研究レベルでは進んでいる。
- 改善案: (1) V5〜V9の比較を、単発の`baseline=0.950`のようなローカル評価ではなくrepeated CVでのノイズフロア測定（8/4のGLOSSARY参照）と組み合わせ、差が本当に意味があるかを検証する。(2) 8/10notebookの「ローカル関連性ランカー」のような後処理の追加。(3) TTAの8パターン全てが本当に効果があるか、パターンごとの寄与を切り分けて計測する。(4) end-to-end方向の研究動向を踏まえ、検出とリンク予測の同時学習も中長期的な検討候補。

## 2. Predicting Smartphone Addiction (Playground S6E8): S6E8 | Smartphone Addiction Ceiling Breaker

- **コンペ**: [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)（Playground Series Season 6 Episode 8）
- **原著者**: ANHAD MAHAJAN ([anhadmahajan06](https://www.kaggle.com/anhadmahajan06))
- **元notebook**: [S6E8 | Smartphone Addiction Ceiling Breaker](https://www.kaggle.com/code/anhadmahajan06/s6e8-smartphone-addiction-ceiling-breaker)
- **スコア**: Public/Best Score **0.97090**

**学べる主要テクニック**
- 複数の公開submission CSVをファイル名のスコア表記から自動収集する仕組み
- スコアを4乗して重みとする「動的べき乗重み付け」（最高スコアモデルへの重み集中）
- ランク変換 → logit変換 → 重み付き平均 → expit逆変換という「非線形ブレンド」（Strategy 1: Logit Blending）
- 90%/10%の極端アンカーブレンド、べき乗によるシャープネスブレンド、Min-Maxブレンドという4戦略の比較出力

**評価指標の要約**: ROC-AUCは予測値の絶対値ではなく順位のみで決まるため、このnotebookは一貫して「ランクに変換してから統合する」設計を取っている。特にlogit変換は0.999と0.9999のような裾（tail）の差を引き伸ばし、ほぼ同じ内容の予測同士でも並び順に差をつけられるようにする狙いがある。

**改善点の考察**
- 同コンペのBest Score上位には、本notebookと同じ「複数の公開submissionをブレンドする」タイプが多いが、中には学習コード自体を非公開にした「提出専用notebook」（GLOSSARY参照）もあり、そうしたnotebookはブレンドの元になった個々のモデルの多様性を検証しづらいという弱点がある。本notebookは4つの異なる非線形変換戦略を透明に比較できる点で学びやすい。
- 2025年のKaggle Playgroundコンペの傾向調査では、単純なブレンドより「3層構造のスタッキング」（レベルごとの予測を次のレベルの入力にする）が優勝解法で使われた例が報告されており、OOF予測が入手できるならメタモデルによるスタッキングの方が理論上は高い天井を持つ可能性がある。
- 改善案: (1) 本notebookはOOF予測ではなく最終submission予測のみをブレンドしているため、Hill Climbing（8/3のGLOSSARY参照）やロジットスケールでのスタッキング（8/4のGLOSSARY参照）のような、OOFに基づく重み最適化に置き換えられればさらに改善余地がある。(2) 4戦略の相対的な優劣が「提出して確認する」前提になっており、CVでの事前評価ができない設計になっている点は、直接the metric（=順位相関）を使ったオフライン検証に置き換える余地がある。(3) 元にしているモデル群の多様性（GBM系だけか、TabM等の深層表形式モデルを含むか）を明示すればブレンドの効果予測がしやすくなる。

## 3. RSNA Knee Abnormality Detection: exp-2

- **コンペ**: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- **原著者**: PRITOM2357 ([pritom2357](https://www.kaggle.com/pritom2357))
- **元notebook**: [exp-2](https://www.kaggle.com/code/pritom2357/exp-2)
- **スコア**: Public/Best Score **0.899**

**学べる主要テクニック**
- DICOMヘッダから撮像条件（脂肪抑制・T1/T2/PD強調）と左右（laterality）を、CSVラベルに頼らず幾何学的・タグベースで復元
- ミリメートル単位の物理サイズクロップと、患者座標系への投影によるスライス順序の幾何学的復元
- ラベル（診断項目）ごとに専用クエリを持つSlotHeadでのスロット別Attention（8/7のGLOSSARY「ターゲット別クエリAttentionヘッド」と同系統）
- モデルの指紋照合（fingerprint）によるロード時の前処理整合性チェックと、デュアルGPUワークキュー＋フェイルセーフ部分保存によるアンサンブル推論
- ターゲット（所見）ごとに異なるTTAプーリング戦略（局所所見はmax/top-k、広範囲所見は平均）

**評価指標の要約**: 12種類の所見それぞれについてROC-AUCを計算しマクロ平均する評価と考えられる（8/6のGLOSSARY「マクロ平均AUC」参照）。本notebookは予測値をパーセンタイル順位に変換してからアンサンブルする設計で、この「順位のみが評価される」性質に一貫して最適化されている。

**改善点の考察**
- 同コンペのBest Score上位（0.90台）には、画像モデルに加えて放射線科レポートのテキストをルールベース/LLMで解析し弱教師あり学習の代理ラベルとして使う「レポート由来の弱教師あり学習」（8/10のGLOSSARY参照）を採用したnotebookがあり、本notebookは画像のみのアンサンブルに留まっている点で、テキスト情報を活用する余地が残っている。
- 2025年の研究では、DINOv2のような自己教師あり基盤モデルを3D医用画像（膝MRIを含む）に適応させる"Medical Slice Transformer"のような手法や、OAIデータでの自己教師ありDINO事前学習が、ゼロからの学習やImageNet事前学習を上回る性能を示しており、本notebookが使う自然画像で事前学習したDINOv2-smallをそのまま流用する構成は、医用画像ドメインに特化した継続事前学習（continued pretraining）でさらに伸びる可能性がある。
- 改善案: (1) レポートテキストを弱教師ラベルとして併用し、画像単独モデルとのアンサンブルに組み込む。(2) DINOv2をOAI膝MRIデータ等で継続事前学習してからfine-tuningする。(3) TTAのターゲット別プーリング戦略（max/top-k vs mean）の選択根拠が明記されていないため、各所見でどちらが効くかをholdoutで検証してから固定する。(4) デュアルGPUワークキューの並列化ロジックは推論速度改善に有効だが、モデル数が増えた際のロード時間（`BUILD_LOCK`のシリアライズ）がボトルネックにならないか計測する余地がある。

---

## まとめ

3本とも「単一の派手な新モデル」ではなく、**しきい値・重み・後処理・データの読み方といった地味だが効く部分を丁寧に詰める**アプローチが共通していた。特にRSNA notebookのDICOMヘッダからの左右・撮像条件復元は、医療画像特有の「タグの欠損」にどう向き合うかという実務的な学びが大きい。
