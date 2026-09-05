# 週次レビュー 2026-08-30 〜 2026-09-05

今週の日次レビュー5日分（08-30〜09-03、計15本）を横断して読み直した記録。
**点で読んだ15本を、線として繋ぎ直すこと**が目的。

一言でまとめるなら、今週は **「測定器を作る週」** だった。

先週（08-23〜08-29）は「混ぜ方の週であり、同時に混ぜるのをやめる基準を探した週」だった。
その「基準」が、今週は具体的な**計測装置**として3コンペすべてに実装された形で現れる。

- 08-30「指標をよく読む——同じ行為が、設計指針にもLB過剰適合にもなる」
- 08-31「改善が本物かノイズかを判定できる検証設計を先に用意すること」
- 09-01「著者がそのスコアをどれだけ疑っているかを見るべき」
- 09-02「効いたと言うために何を測るかを先に決めている」
- 09-03「効いたを主張する前に、効いたことを見える形にする」

5日間の見出しが、ほぼ同じことを別の言葉で言っている。
これは偶然ではなく、**公開notebookが軒並み天井に張り付いた結果、差がつく場所が
「手法」から「実験の管理方法」へ移動した**ことの反映である。

そして今週最大の発見は、その帰結として **改善提案の第1位が3コンペとも
「積み上げた要素を個別に評価（ablation）せよ」に収束した**こと。
要素を足すのは簡単で、効いているかを測るのは面倒——この非対称性が、
公開notebookが天井に張り付く一番の理由だと、5日かけて確認した週だった。

---

## 1. 対象期間と扱ったコンペ一覧

対象は直近の日曜〜土曜（2026-08-30 〜 2026-09-05）。
稼働は **08-30（日）〜09-03（木）の5日間・計15本**。09-04（金）は稼働なし、09-05（本日・土）はこの週次まとめの実行日。

今週の構造的な特徴は、**Playground枠でコンペの世代交代が週内に起きた**こと。
S6E8（Predicting Smartphone Addiction）が 09-01 に終了し、09-02 から S6E9（Predicting Electric Vehicle Purchases、9/1開始・9/30終了）に切り替わっている。
S6E8 は本ルーティンで20本以上扱った長期銘柄で、その**最終日までを追い切った**のは今週の収穫。

| 日付 | Biohub - Cell Tracking | Playground | RSNA Knee Abnormality Detection |
|---|---|---|---|
| [08-30](../../Daily_Kaggle_Review/2026-08-30/) | [Biohub Cell Tracking 92.6%](https://www.kaggle.com/code/salemali7/biohub-cell-tracking-92-6) **0.926** | S6E8: [🚨 OVERFITTING TRAP - Do Not Copy](https://www.kaggle.com/code/najiama/overfitting-trap-do-not-copy) **0.97129** | [RSNA Knee DINO Protocol Fusion](https://www.kaggle.com/code/llccqq624/rsna-knee-dino-protocol-fusion) **0.935** |
| [08-31](../../Daily_Kaggle_Review/2026-08-31/) | [Biohub 0.933 LB](https://www.kaggle.com/code/evgendvorkin/biohub-0-933-lb) **0.933** | S6E8: [smartphone addiction eda fast](https://www.kaggle.com/code/lamhuy8904/s6e8-smartphone-addiction-eda-fast) **0.97125** | [RSNA Baseline](https://www.kaggle.com/code/evgendvorkin/rsna-baseline) **0.936** |
| [09-01](../../Daily_Kaggle_Review/2026-09-01/) | [c35-fallback-detection-cleanroom](https://www.kaggle.com/code/tangai1/biohub-c35-fallback-detection-cleanroom-20260831) **0.934** | S6E8（最終日）: [Rank-Gauss Stack](https://www.kaggle.com/code/asteriosterzis/predicting-smartphone-addiction-rank-gauss-stack) **0.97130** | [Knee MRI: twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) **0.924** |
| [09-02](../../Daily_Kaggle_Review/2026-09-02/) | [Biohub 0.934 LB PROXY_SCORE=0.9384](https://www.kaggle.com/code/evgendvorkin/biohub-0-934-lb-proxy-score-0-9384) **0.934** | **S6E9**: [EV Adoption Tokens: XGBoost + TinyTokenTransformer](https://www.kaggle.com/code/chovyxu/ev-adoption-tokens-xgboost-tinytokentransformer) **0.94539** | [RSNA Knee Abnormality DetectionV1](https://www.kaggle.com/code/kunaldesale2408/rsna-knee-abnormality-detectionv1) **0.936** |
| [09-03](../../Daily_Kaggle_Review/2026-09-03/) | [Agreement-Gated Dual-Seed Fusion](https://www.kaggle.com/code/flexonafft/biohub-agreement-gated-dual-seed-fusion) **0.935** | S6E9: [S6E9 LightGBM](https://www.kaggle.com/code/kirill0212/s6e9-lightgbm) **0.94607** | [RSNA Knee: Full 4-arm ensemble v55](https://www.kaggle.com/code/nishantkharga/rsna-knee-full-4-arm-ensemble-v55) **0.935** |

### 週を通したスコアの動き

- **Biohub**: 0.926 → 0.933 → 0.934 → 0.934 → 0.935。
  **先週の週次レビューで「正攻法帯の天井は 0.927」と記録した数字が、今週 0.935 に上がった**（+0.008）。
  上げた主因は明確で、**dual-seed融合 + 調和平均による双方向エッジ統合 + frame retention guard** の3点セットが
  08-31以降のすべてのnotebookに入っている。0.926 の 08-30 だけがこの構成を持たない。
  一方で LB全体の上位 0.95〜0.966 帯は今週も**メトリックハック系**であることが繰り返し確認され、
  「honest な公開上限 ≈ 0.935 / LB1位 0.963」という位置づけが 09-03 に明文化された。

- **Playground**: S6E8 が 0.97129 → 0.97125 → 0.97130 で終了。**3日間の変動幅 0.00005**。
  09-01 の Rank-Gauss Stack が、**public LB（59,260行）の分解能を実測して約 0.00007** と示した。
  つまり**上位陣が争っていた差は、測定分解能より小さかった**ことが、コンペ最終日に定量的に確定した。
  S6E9 に移ると 0.94539 → 0.94607 で、こちらは開始直後にもかかわらず既に上位3本が幅 0.0003 に密集している。
  **飽和のパターンが、新コンペで2日目から再生産されている**のが今週最も示唆的な観察。

- **RSNA Knee**: 0.935 → 0.936 → 0.924 → 0.936 → 0.935。
  09-01 の 0.924 は「単一モデル・アンサンブルなし・TTAなし・実行53秒」で、**意図的に低い方を選んだ**日。
  4アーム30モデル超（18分38秒）との差がわずか 0.012 しかないことが、この1本で可視化された。
  09-03 の v55（0.935）は元レシピ（0.936）の再組み立てで **0.001 落ちている**点も記録に値する。

---

## 2. 今週繰り返し登場した手法・パターン Top5

### ① 「改善が本物か」を判定する計測器を先に作る — 15本中13本、3コンペすべて

今週の中心テーマ。同じ目的に対し、コンペごとに違う実装が出そろった。

| 実装 | 登場 | 何を測るか |
|---|---|---|
| **PROXY_SCORE（公式指標のローカル再実装）** | Biohub 09-02 / 09-03 | 提出枠を使わずに変更の良し悪しを測る。ハンガリアン法の7µmゲート照合・Edge Jaccard・ノード数ペナルティ・Division Jaccard を全部自前で実装 |
| **public LB分解能の実測** | S6E8 09-01 / S6E9 09-02・09-03 | 相関0.99の2提出がノイズだけで生む差（59,260行で約0.00007）。**争っている差がこれ以下なら判定不能** |
| **ノイズフロアの実測** | S6E9 09-03 | 無意味なランダム列を足したときのAUC変動幅。改善の下限を確定させる |
| **2000回ブートストラップでの優位判定** | RSNA 09-01 | 学習データ拡張 0.8923→0.9054 が「92.7%の試行で優位」。**差の大きさではなく信頼性で採否を決めている** |
| **イベント認識のバリデーション標本設計** | Biohub 09-01 / 09-02 | 分裂がGTに注釈された動画は約44%しかない。GTグラフを開いて「出次数2以上のノードがあるか」を確認し、**測りたい現象を含まない検証セットを構造的に排除する** |

**なぜ効くか**: 3コンペとも公開帯が飽和し、差が小数第3〜5位に収まっている。
この領域では「上がった/下がった」の観測は情報を持たず、**「有意に上がったか」しか意味がない**。
計測器を持たない者は、ノイズを実力と誤認して積み上げ続けることになる。

最も鮮やかだったのは **RSNA 09-01**。45症例のゴールドセットで3本ブレンドが +0.010 を示したが、
実LBでは 0.914 → 0.914〜0.915 にしかならず、著者は「ゴールドセットのノイズだった」と結論して**アンサンブルを捨てた**。
一方でブートストラップ92.7%優位を確認できたデータ拡張は採用した。**測ったうえで捨てる**という判断が、今週最も学ぶ価値のある態度だった。

### ② 順位（rank）空間での融合 — AUCコンペ10本すべて

ROC-AUC は**順位のみに依存し単調変換で不変**。この性質を融合設計に直結させるパターンが完全に定着した。

- `rank(pct=True)` によるパーセンタイル順位化（RSNA 08-30・09-02・09-03、S6E8全日）
- **Rank-Gauss変換**（順位 → `norm.ppf` で標準正規へ）— S6E8 09-01。外れ値の影響を消したまま線形メタモデルが好む形にする
- `argsort(0).argsort(0)` による順位化 — RSNA 09-03。fold平均の**前に**順位化して各foldを対等にする
- `prob` / `rank01` / `logit_rank01` の**3ビュー**で貪欲ブレンドを回す — S6E9 09-02

**なぜ効くか**: スケールもキャリブレーションも桁違いのモデル（生スコアが数十のオーダーのものを含む）を、
**無調整で対等に混ぜられる**。順位不変性がある以上、順位空間で混ぜてもスコアは1ビットも損なわれない。

同時に今週は、この性質の**裏面**もはっきり示された。
09-02 RSNA の `consensus power calibration (γ=1.05)` は**単調変換なので理論上 AUC を1ミリも動かさない**のに、
焦点4ターゲットにだけ適用されていた——これは LB への過学習の痕跡である可能性が高い、という指摘。
**「順位不変だからキャリブレーション単体では原理的に上がらない」を逆に使った異常検知**として応用が効く。

### ③ 調和平均による双方向エッジ融合 + retention guard — Biohub 5本中4本

Biohub の正攻法帯を 0.927 → 0.935 に押し上げた実体。3つの部品がセットで動く。

1. **Harmonic bidirectional fusion**: リンクの確信度を t→t+1 と t+1→t の両方向で計算し、**調和平均** `2ab/(a+b)` で統合。
   調和平均は小さい方に強く引かれるので、**片方向だけが自信を持つ「片思いリンク」（=FP候補）が狙って落ちる**。
2. **Agreement gate**: 異なる乱数シードの検出器2本を同じ調和平均で融合し、「両方が同意した検出だけを強い候補にする」フィルタにする（09-03）。
3. **Frame retention guard**: 融合後の候補ノード数が主モデル単独の90%を下回ったフレームは、**丸ごと主モデルに巻き戻す**。

**なぜ効くか**: 指標が Jaccard `TP/(TP+FP+FN)` なので **FPとFNの両方が罰される**。
①②はFPを削る方向、③はFNを増やさない方向で、**指標の2つの分母成分にそれぞれ別の道具を当てている**。
特に③は「アンサンブルは平均的には良いが、局所的には悪化しうる」という現実への安全弁で、
08-31 のnotebookでは v2→v4 の改善 +0.0035 の主因と明記されている。

**残った疑問**: 調和平均は一般化平均（power mean）の p=−1 の特殊ケースにすぎない。
p を検証セットで探索した形跡がどのnotebookにもなく、**p=−1 が最適である根拠はまだない**（09-01 の改善提案②）。

### ④ 静かな失敗を、大きな音の失敗に変える — 15本中9本

先週から継続しているテーマだが、今週は道具立てが完成形に近づいた。

- **Configuration Guard**: 重要パラメータを期待値表と照合し、1つでも違えば実行**前**に `RuntimeError`（fail fast）
- **Pipeline Manifest**: 設定値ではなく**実際にロードされたか**を `Path.exists()` と `globals()` で観測して印字。
  `requested=True, loaded=False` を明示的に警告する。09-03 の原著者いわく「これまで見つかった静かなバグは全部この印刷ひとつで見えたはずだった」
- **label-free audit**: 正解を使わず、スキーマ・ID連番・グラフのトポロジー（1親→3子がないか）だけで submission を検証
- **モンキーパッチの assert**: 上流スクリプトを文字列置換で書き換える際、置換対象が見つからなければ即エラー。
  これを欠くと「8視点TTAのつもりで4視点で走る」という静かな劣化が起きる（09-02 の改善提案④）
- **重みファイルの SHA-256 検証**: 添付データセットが黙って差し替わる事故を検出（RSNA 09-03）
- **fail-open / fail-closed の使い分け**: 提出形式の検証は「壊れていたら止める」、任意の改善処理は「失敗したらスキップ」（RSNA 08-30）

**なぜ効くか**: 31分〜5時間のGPU推論を無駄にしないための投資であると同時に、
**「実行できた = 設定は意図どおり」という暗黙の仮定を明示的に検査する**行為でもある。
①の計測器と同じ思想が、実行環境側に向けられたもの。

### ⑤ LLMによるレポート→ソフトラベル生成（弱教師あり学習） — RSNA 5本中4本

このコンペの構造そのもの。**4,407 study のうち構造化ラベルが付いているのは58件だけ**で、
残りは自由記述の読影レポート（12言語）しかない。

- レポートを言語モデルに読ませ、12所見の**確率**に変換（「断裂が疑われる」→ 1ではなく **0.8**）
- これで学習データが 58 → 4,349 に（**75倍**）
- **58件は一切学習に使わず検証専用**にする規律（そこでの macro AUC 0.9167）

**なぜ効くか**: 確信度の差をそのままラベルに残せるので、0/1に丸めるより情報を失わない。
そして今週の観察として決定的なのは、**公開帯が 0.935〜0.936 で飽和しているのは、全員が同じ弱教師ラベルを使っているから**という仮説。
つまり **画像モデルを増やすより、ラベルの質を上げる方が伸びしろが大きい**。
多言語の否定表現（「ACL断裂なし」「no evidence of tear」）の検出精度を上げるだけで、稀な所見のラベルノイズが直接減る。

**同時に露呈した弱点**: 58件のホールドアウトで貪欲前向き選択を何十回も回すのは**検証セット過学習そのもの**。
稀な所見の陽性が数件しかないため macro AUC は ±0.02 程度は普通に動く。
09-01 の著者は 45症例での判断が誤りだったことを自ら認めている。

---

## 3. 今週の珍しい・特徴的な手法

### ⑴ 意図的な「反面教師」notebook — Reverse Micro-Sorting（08-30 / S6E8）

タイトルが「🚨 OVERFITTING TRAP - Do Not Copy」。原著者自身が「これは選ばない」と宣言したうえで、
**public LB 0.97129（Rank #22相当、公開notebookの最前線）**を出している。

手口はこう。ベースラインの順位を `pd.qcut(q=500)` で500バケットに分け、
`np.lexsort((id, -lgbm_rank, bucket))` で**バケット内だけ**を自前LGBMの順位の**符号を反転して**並べ替える。
大域順位は保たれるのでAUCの本体は壊れず、微細な順位だけが Public LB の20%サンプルに適合する。得点 **+0.00001**。

これが教材として優れているのは、**「順位不変性」という②で見た正しい性質が、そのまま悪用の入口になる**ことを実演している点。
09-01 で実測された分解能 0.00007 と並べると、**+0.00001 は分解能の 1/7** であり、
この notebook が得ていたものが何だったかが数字で確定する。

理論的裏付けは [The Ladder (Blum & Hardt, ICML 2015)](https://proceedings.mlr.press/v37/blum15.pdf)。
「提出のたびにLBスコアを見て次を決める」適応的な手続きがなぜ統計的保証を壊すかを定式化し、
**前回より有意に良いときだけスコアを更新する**Ladderメカニズムを示している。
このnotebookの行為は、まさにLadderが弾くために設計されたパターンそのものだった。

### ⑵ 表の1行を「小さな言語」に翻訳するトークン設計（09-02 / S6E9）

`EV Adoption Tokens: XGBoost + TinyTokenTransformer`。
数値フィールド1つ = 1トークン、カテゴリ1つ = 1トークンとし、トークンの中身に
**ロバスト標準化値・欠損フラグ・フーリエ基底 sin/cos(f=1,2,4,8)・局所ガウス基底・分位ビン頻度・丸め値頻度**を詰め込む。
同じ情報から **XGBoost用の平坦行列**と **Transformer用のトークンテンソル**という2ビューを作る。

これは [FT-Transformer](https://openreview.net/pdf?id=i_Q1yrOegLY) / [On Embeddings for Numerical Features (arXiv:2203.05556)](https://arxiv.org/pdf/2203.05556) の
**PLR埋め込み（周期的埋め込み＝まさにフーリエ基底）の実装版**。論文の主張は「数値特徴に高度な埋め込みを与えると、深層表形式モデルとGBDTの差が埋まる」。

**ただし費用対効果は厳しい**。同コンペの [Single XGB](https://www.kaggle.com/code/evgendvorkin/single-xgb) が **0.94327** を出しており、
この重厚なパイプライン 0.94539 との差は **0.002 しかない**。①の計測器の話に戻るが、
ablation なしにこれを「効いた」と言うことはできない。

### ⑶ 桁分解特徴量（digit features）（09-03 / S6E9）

`(x // 10**k) % 10` を k = −4..3 で作り、13列 → **+104列**。
合成データの生成器が残す「丸め幅・格子の指紋」を、木モデルが**1回の分割で使える形**にする。
木は周期的・非単調な構造の表現が極めて苦手なので、明示的に与える価値が大きい。

08-31 の S6E8 で出てきた**小数桁特徴（Decimal Lattice）** `frac_x = x - floor(x)` / `d1_x = floor(x*10) % 10` と同系統で、
**今週このファミリーが2つのPlaygroundコンペにまたがって登場した**のは注目に値する。

理論的背景は [On Privacy Leakage in Tabular Diffusion Models](https://arxiv.org/html/2605.06835v1) が示す
「合成データは有用性が高いほど元データへの過学習（＝痕跡の残存）が起きやすい」というトレードオフ。
ただしこれは**汎化する知識ではなく生成器の癖の暗記**なので、Kaggleが分割方法を変えていれば消える。ablation必須。

---

## 4. 今週のGLOSSARY新規追加語（33件）

詳細は [GLOSSARY.md](../../GLOSSARY.md) を参照。今週は特に**検証・診断まわりの語彙**が厚くなった。

**検証・計測（今週の中心）**

- 最小検出可能差（Minimum Detectable Difference / リーダーボードの分解能）— 09-01
- 公式指標のローカル再実装による提出不要プロキシ（Submission-Free Proxy Score）— 09-02
- 検証セット過学習（Validation Overfitting / 適応的過学習）— 08-31
- 単調変換の無効性チェック（Monotone-Transform No-Op Check）— 09-02
- fold別AUCとプールドOOF AUCの乖離（Fold-wise vs Pooled OOF AUC Divergence）— 09-03
- エラーの成分分解（Error Decomposition）— 09-01
- 単一ノブ掃引（One-Variable-at-a-Time / Single-Knob Sweep）— 09-03
- 上限バインド診断（Cap-Binding Diagnostics）— 09-03

**融合・アンサンブル**

- バケット内マイクロソート（Bucketed Micro-Sorting）— 08-30
- 負の重みブレンド（Negative-Weight Blending）— 08-30
- 不確実性マスキング（Uncertainty Masking）— 08-30
- 撮像プロトコル別スロット融合（Multi-Protocol Slot Fusion）— 08-30
- 候補保持率の下限（Minimum Candidate Retention Floor）— 09-03
- ノード数ペナルティ付きJaccard（Adjusted Jaccard with Node-Count Penalty）— 09-01

**特徴量エンジニアリング**

- 小数桁特徴量（Fractional / First-Decimal-Digit Features）— 08-31
- 合成データのフォレンジック特徴量（Synthetic-Data Forensic Features）— 09-02
- マルチスケール分位ビン・ターゲットエンコーディング — 09-02
- ベイジアン平滑化つきターゲットエンコーディング — 08-31
- 上三角マスクによる完全相関列の除去 — 09-03
- ハイパーパラメータの特徴量化（Hyperparameter-as-Feature）— 09-03
- [CLS]トークンによる行集約 — 09-02
- スペクトルバイアス（Spectral Bias / Frequency Principle）— 09-02

**運用・エンジニアリング**

- フェイルオープン / フェイルクローズドの使い分け — 08-30
- モデル・スイッチボード（Model Switchboard with Frozen Folds）— 09-03
- 移植ルールの再キャリブレーション（Ported-Rule Re-calibration）— 08-30
- アームの逐次実行によるピークRAM抑制 — 09-01
- 競技データツリーの再帰globの禁止 — 09-01
- PhotometricInterpretation と MONOCHROME1 の反転（DICOMの輝度極性）— 09-01

**モデル・アルゴリズム**

- Higher-Order Cell Tracking Transformer（HOCT）— 08-31（文献経由）
- 微分可能最適化層（Differentiable Optimization Layer, DOL）— 09-02（Paper Digest経由）
- SLSQP（逐次二次計画法）— 08-31
- CTGAN（Conditional Tabular GAN）— 08-31

---

## 5. 今週の論文消化との接続

[Paper_Digest 2026-09-02](../../Paper_Digest/2026-09-02/) — **Differentiable optimization layers enhance GNN-based mitosis detection**
（Zhang, Nguyen, Tsuda / Scientific Reports 13:14306, 2023 / [DOI](https://doi.org/10.1038/s41598-023-41562-y) / [コード](https://github.com/95-HaishanZHANG/GNN-DOL)）

この選定は、**日次レビューで見つけたボトルネックに論文を当てにいった**という点で今週の白眉。

Biohub の日次レビュー（08-30 / 08-31 / 09-01）で繰り返し記録されていたのは、
**Division Jaccard がほぼ機能していない**という事実だった。08-31 の validation 実測で
`Division Jaccard = 0.1667 / Division FP = 0` ——**FPは0だがTPも極端に少ない = 保守的すぎる**。
分裂は指標の約15%を占めるのに、そこがまるごと死んでいる。

GNN-DOL はこの問題に**モデル側**から答える。
**微分可能最適化層（DOL）**は「数理計画問題そのものをニューラルネットの1層として使う」技術で、
逆伝播はKKT条件と陰関数定理から解析的に通す。普通の層と違い、**出力にハード制約を強制できる**。

```
z* = argmin (1/2) zᵀ Q z
     s.t.   A₁z = 1     ← t+1の各細胞は、tのちょうど1細胞に対応（娘は必ず親を持つ）
            A₂z ≤ 2     ← tの各細胞は、t+1の高々2細胞にしか対応しない（分裂は2分裂まで）
            0 ≤ z ≤ 1
```

**Kaggle上の実践との接続点は3つ**。

1. **後処理でやっていることを、学習に移せる**。今週の Biohub notebook はどれも
   `SAFE_DIV_*` の距離しきい値・`DEEPCENTER_SAFE_DIV_VETO` の閾値といった**手組みルール**で分裂を判定している。
   09-03 の notebook は `GAP_CLOSE_UM=5.8` のような定数が20個以上並んでおり、手動チューニングの上限に達している。
   DOL は「1親→高々2娘」という同じ制約を、**ILPの外の後処理ではなくモデル内の層として**課す。

2. **「分裂が同時に2件以上起きるケースは訓練データにほとんど現れない」問題への答え**。
   著者らはここに「既知の制約を明示的に埋め込む」動機を見出している。
   これは Biohub の「分裂がGTに注釈されている動画は約44%しかない」という状況と完全に相似形で、
   **データが足りない領域ではドメイン知識が正則化として働く**という、RSNA の `SLOT_PRIOR_TABLE`
   （ACLは矢状断、MCLは冠状断とハードコード）とも同じ構造をしている。

3. **エッジ同士の類似度を Q に入れている**点が、今週 Biohub で3日にわたって参照された
   [HOCT (arXiv:2607.11754)](https://arxiv.org/abs/2607.11754) の「エッジ中心アーキテクチャ」と方向が一致する。
   現行 notebook の bidirectional fusion は「1本のエッジについて2方向を見る」に留まり、**エッジ間の相互作用は見ていない**。
   GNN-DOL も HOCT も、そこに二次の（構造的な）情報があると言っている。

---

## 6. 一言まとめ

今週で、**「疑う」から「測る」への移行が完了した**と言っていい。

先週の週次レビューは「疑うための道具が3コンペにわたって出そろった」で終わっていた。
今週はその道具が実際に判断を変えた事例が出た——RSNA 09-01 の著者は、
ゴールドセットで +0.010 を示したアンサンブルを**測ったうえで捨て**、実行時間を1/3にした。
S6E8 09-01 の著者は、自分が公開最高スコアを持ちながら、**その差が測定分解能以下であることを計算で示した**。

一方で、5日間の改善提案の第1位が3コンペとも「ablation を取れ」になったことは、
**測定器を持っている人ですら、自分が積み上げたものには測定器を向けていない**ことを意味する。
09-03 の Biohub notebook は「変更は1点だけ」と宣言しながら、
gap closing・short track rescue・dual-seed融合・DeepCenter veto・bidirectional融合の5機構を同時に有効化している。
**公式指標のオフライン検証器を持っているのに**、である。

### 来週意識したいテーマ

1. **自分の実験に ablation を義務づける**。要素を1つ足したら、それを抜いた実行を1回する。
   今週見た15本すべてに欠けていた習慣で、そこが公開notebookの天井の正体だった。
2. **新しいコンペ（S6E9）の飽和曲線を最初から観察する**。開始2日目で既に上位3本が幅0.0003に密集している。
   **飽和する前の段階を見られる**のは今回が初めてなので、いつ・どうやって収束が起きるかを記録する価値が高い。
3. **賞金トラックの空白を狙う視点**。RSNA の Efficiency Prize（$18,000、`Efficiency = AUC項 + Runtime/32400`）は
   今週3日にわたって「既存資産だけで挑戦できるのに誰も本気で狙っていない」と指摘された。
   **精度トラックが 0.936 で3本以上横並びなら、競争の薄い側に移る**という判断は、
   モデルの改善より期待値が高い場面がある。
4. **ラベルの質 vs モデルの数**。RSNA で最も伸びしろが大きいのは5本目のアームではなく、
   多言語レポートの否定表現の検出精度だという仮説が今週固まった。同じ発想は他のコンペにも移植できるはず。

---

## 参考文献・出典

- Kaggle: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) / [Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8) / [Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) / [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [The Ladder: A Reliable Leaderboard for Machine Learning Competitions (Blum & Hardt, ICML 2015)](https://proceedings.mlr.press/v37/blum15.pdf)
- [Higher-Order Cell Tracking Transformer (arXiv:2607.11754)](https://arxiv.org/abs/2607.11754)
- [Differentiable optimization layers enhance GNN-based mitosis detection (Sci Rep 13:14306)](https://doi.org/10.1038/s41598-023-41562-y)
- [On Embeddings for Numerical Features in Tabular Deep Learning (arXiv:2203.05556)](https://arxiv.org/pdf/2203.05556) / [FT-Transformer](https://openreview.net/pdf?id=i_Q1yrOegLY)
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data (arXiv:2506.16791)](https://arxiv.org/html/2506.16791v1) / [TabM (arXiv:2410.24210)](https://arxiv.org/html/2410.24210)
- [PromptRad: Knowledge-Enhanced Multi-Label Prompt-Tuning for Low-Resource Radiology Report Labeling (arXiv:2605.20052)](https://arxiv.org/pdf/2605.20052)
- 日次レビュー: [08-30](../../Daily_Kaggle_Review/2026-08-30/) / [08-31](../../Daily_Kaggle_Review/2026-08-31/) / [09-01](../../Daily_Kaggle_Review/2026-09-01/) / [09-02](../../Daily_Kaggle_Review/2026-09-02/) / [09-03](../../Daily_Kaggle_Review/2026-09-03/)

各notebookの著作権は原著者に帰属します。
