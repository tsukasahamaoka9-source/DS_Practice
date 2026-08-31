# 2026-09-01 Kaggle日次レビュー

本日の3本に共通するテーマは **「自分のスコアをどこまで疑えるか」**。

3本とも、単に手法を積み上げてスコアを上げるのではなく、**「その改善は本物か」を確かめる仕掛け**を持っています。Biohubは実行時の状態を機械が検査し、Playgroundは公開LBの分解能を実測し、RSNAはアンサンブルの効果を測って割に合わないと判断して捨てています。

| 枠 | コンペ | notebook | スコア |
|---|---|---|---|
| 固定 | Biohub - Cell Tracking During Development | [biohub-c35-fallback-detection-cleanroom-20260831](https://www.kaggle.com/code/tangai1/biohub-c35-fallback-detection-cleanroom-20260831) | **0.934** |
| Playground | Predicting Smartphone Addiction (S6E8) | [Rank-Gauss Stack](https://www.kaggle.com/code/asteriosterzis/predicting-smartphone-addiction-rank-gauss-stack) | **0.97130** |
| 実コンペ | RSNA Knee Abnormality Detection | [Knee MRI: twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) | **0.924** |

---

## 1. Biohub - Cell Tracking During Development

**`biohub-c35-fallback-detection-cleanroom-20260831`** — 原著者: TANGAII / Public 0.934 / 実行 32分29秒 (GPU T4 x2)
元notebook: https://www.kaggle.com/code/tangai1/biohub-c35-fallback-detection-cleanroom-20260831

ゼブラフィッシュ胚の3D+時間動画から、細胞を検出し・フレーム間で追跡し・分裂イベントを検出する。**メトリックハック系（0.965帯）を除いた正攻法の公開最高帯**。

### 評価指標の要約

**Edge Jaccard（約85%）+ Division Jaccard（約15%）** の重み付き和。Jaccard は `TP / (TP+FP+FN)` なので、**余計に出した予測も分母に効く**。さらに `estimated_number_of_nodes` を超えたノード数にペナルティがかかるため、「迷ったら出す」が構造的に損になる。ラベルがsparseなため画素ベースの精度は使えず、生物学的価値は「系譜の復元」にあるのでノードではなくエッジで測る。分裂は約44%の動画にしか存在しないので、多数派のリンクに埋もれないよう別成分として明示的に評価される。

### 学べる主要テクニック

- **Frame retention guard**: アンサンブル後の候補数が主モデル単独より減ったフレームは、主モデルへロールバックする。「平均を取ることで確信度がしきい値を下回り、細胞を丸ごと見落とす」というアンサンブル特有の失敗を安全弁で塞ぐ。
- **Harmonic bidirectional fusion**: リンクの確信度を t→t+1 と t+1→t の両方向で計算し、**調和平均**で統合。調和平均は小さい方に強く引かれるので、片方向だけ自信のあるリンク（=FP候補）が狙って落ちる。
- **Configuration guard / Pipeline manifest**: 設定値を実行時にassertで固定し、最後に「要求した状態」と「実際にロードされた状態」を**別々に**出力する。`requested=True, loaded=False` を明示的に警告する設計。
- **Label-free audit**: 正解なしで、スキーマ・データセット網羅性・グラフのトポロジー・guardログの内部整合性を検査する。
- **µm単位での距離計算**: z軸(1.625µm)はxy軸(0.40625µm)の4倍粗い。ボクセル単位のまま距離を測ると近傍判定が歪む。
- **検証セットの層化**: 分裂を含む動画を明示的に選ぶ。ランダムに選ぶと分裂ゼロのサンプルばかりになり、指標の15%が検証で動かなくなる。

### 改善点の考察

**他notebookとの比較（同コンペ上位を確認）**
- `Biohub Harmonic Fusion`（0.933）と `Biohub - Track Your Cells Development`（0.933）は同じ0.93帯だが、本notebookのようなDeepCenter veto（別モデルによる復元ノードの再検査）は持っていない。一方で `biohub-937-sdw80`（0.930）や `Biohub base 0937`（0.931）は**より多くの入力データセット（+10）**を使っており、本notebookは3データセットに留まる。
- `Biohub Competition Solution`（0.965）/ `Biohub Solution`（0.966）は表示上位だが、既知のメトリックハック系のため学習対象から除外している。**本notebookは "cleanroom"（無菌室）と自称している通り、正攻法での上限を探る立場**。

**関連文献**
- [Higher-Order Cell Tracking Transformer (HOCT, arXiv 2607.11754)](https://arxiv.org/abs/2607.11754) — **エッジ中心（edge-centric）** のTransformerで、深い事前学習済み画像エンコーダなしにCell Tracking Challengeで SOTA。「分裂が系譜を絡ませ、ノードを共有するエッジ同士のラベル一致がほぼランダムになる」という構造的問題を、3D幾何事前分布を持つエッジ中心アーキテクチャで解いている。**本notebookが後処理で解いている問題を、モデル側で解こうとしている**点が対照的。
- [Cell-TRACTR (PLOS Comput Biol)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013071) — セグメンテーションと追跡をend-to-endで解くTransformer。本notebookのような「検出→ILP→後処理」の多段パイプラインとは逆の方向性。

**改善提案**
1. **DeepCenter vetoを、復元ノードだけでなく低確信度の通常検出にも広げる。** 現在 veto は gap closing と safe division の結果にしか適用されていない。`DET_THRESHOLD=0.965` 直上の検出（=最もFPになりやすい帯）にも第二モデルの検査を通せば、Edge Jaccardの分母を追加で削れる可能性がある。
2. **Harmonic fusion の統合関数を調和平均から一般化平均（power mean）に変え、指数 p を検証セットで探索する。** 調和平均は p=-1 の特殊ケース。p を -2 まで下げれば片方向FPをさらに強く落とせるし、-0.5 なら緩められる。現在は p=-1 が最適である根拠がない。
3. **エラー分解（`decompose_errors`）の結果を、パラメータ探索に直結させる。** 「失点の何%が検出漏れ由来か / 誤接続か」を計算する仕組みは既にあるのに、しきい値は手動で固定されている。分解結果を目的関数にした自動探索（例: FN由来が支配的なら `DISAPPEARANCE_WEIGHT` を上げる）に繋げられる。
4. **エッジ中心Transformer（HOCT系）でILPのコスト行列を置き換える検討。** 現在のエッジコストはUNet+Node Transformerのペアワイズ確信度だが、HOCTが指摘する通り「ノードを共有するエッジ同士は独立ではない」。高次の相互作用を入れたコストにすれば、後処理のsafe-division判定を減らせる可能性がある。
5. **入力データセットの拡充。** 0.93帯の他notebookが10以上のデータセット（=より多くの学習済み重み）を使っているのに対し本notebookは3つ。dual-seedを3〜4 seedに広げるだけでも、frame retention guardという安全弁がある以上、リスクは限定的。

---

## 2. Playground Series S6E8 — Predicting Smartphone Addiction（本日終了）

**`Predicting Smartphone Addiction | Rank-Gauss Stack`** — 原著者: Asterios Terzis / Public 0.97130（公開最高帯） / 実行 17分43秒
元notebook: https://www.kaggle.com/code/asteriosterzis/predicting-smartphone-addiction-rank-gauss-stack

**本ルーティンで扱ったS6E8のnotebookは本日で20本を超えたが、その大半は「他人の提出CSVをランク平均するだけ」だった。本notebookは公開最高スコアを出しながら、自分のスコアを疑うことに大半の紙面を使っている。**

### 評価指標の要約

**ROC AUC**。「ランダムな陽性例のスコアがランダムな陰性例より高い確率」。**順位のみに依存し、単調変換で不変**。クラス不均衡でaccuracyが機能せず、かつ運用しきい値が事前に決まらない問題設定に適合する。本notebookはこの性質を徹底活用し、全メンバーをパーセンタイル順位に変換してから統合するため、キャリブレーションが桁違いに異なるモデル（生スコアが数十のオーダーのものを含む）を無調整で混ぜられる。

### 学べる主要テクニック

- **自己参照メンバーの除去**: 公開OOFの中には「モデル」ではなく「このプール自体にフィットされたブレンド」が混じっている。そのOOFはin-sampleで楽観的なので、メタモデルが過大な重みを与え、testで再現しない。**「公開されているOOFだから安全」ではなく、そのOOFがどう作られたかを問う必要がある。**
- **重複メンバーの検出**: 同じ配列が別名で複数データセットに公開されている。重複は無駄なのではなく、**意図しない2倍の重みを静かに導入する**。
- **Rank-Gauss変換**: 順位 → `norm.ppf` で標準正規へ。外れ値の影響を消したまま線形モデルが好む形にする。
- **収束していないフィットは真値より「高く」読める**: `StandardScaler` なしだと lbfgs が収束せず、その結果は実力を過大評価する。`ConvergenceWarning` を無視してはいけない具体例。
- **public LBの分解能の実測**: 59,260行の public に対し、相関0.99の2提出はノイズだけで **約0.00007** 差が出る。上位陣が争っている差（0.97125 vs 0.97130 = 0.00005）は**その分解能より小さい**。
- **単調制約が逆効果になった実例**: スクリーンタイムに単調制約を入れたら 0.0017 悪化。上位3カラムは単調信号を持たず、値ごとの非単調な効果が半分割で r>0.94 再現する構造だった。制約がそれを平滑化して捨てた。

### 改善点の考察

**他notebookとの比較（同コンペ上位を確認）**
- `[S6E8] Top 20 Formula: Dual Master Rank Blend`（0.97128）、`S6E8 Regime-Calibrated Rank Fusion`（0.97127）、`S6E8: Elite Rank Average Ensemble`（0.97126）——**上位帯は全て0.9712〜0.9713に密集しており、本notebookが示した分解能0.00007を考えると、これらは統計的に区別できない**。本notebookが採用していない手法として、regime-calibrated fusion（サンプルを領域分割して領域ごとに重みを変える）があるが、本notebookの分析に照らせばこれは分解能以下の差を追う行為である可能性が高い。
- 一方 `S6E8 | ResNet + FE`（0.96704）や `Predicting Smartphone Addict | NN Residual Network`（0.97129）は、他人の提出に依存せず**自前で学習している**点で本notebookのプールに多様性を足しうる。本notebookのプールは全て公開OOF由来なので、真に独立した新規メンバーが不足している。

**関連文献**
- [The Ladder: A Reliable Leaderboard for Machine Learning Competitions (Blum & Hardt, ICML)](https://proceedings.mlr.press/v37/blum15.pdf) — **本notebookの主張の理論的裏付け**。公開リーダーボードへの適応的な問い合わせを繰り返すと、ホールドアウトの統計的保証が壊れる。Ladderメカニズムは「前回のベストを有意に上回ったときだけスコアを更新する」ことで、問い合わせ回数によらず汎化を保つ。本notebookが手作業でやっている「差が分解能以下なら動かない」という規律を、機構として実装したもの。
- [shakeup (Kaggleリーダーボードのshake-up指標)](https://github.com/davidthaler/shakeup) — public/private順位の差を定量化するツール。本notebookのセクション11（過去7エピソードのバックテスト）と同じ発想。

**改善提案**
1. **自前で学習した独立メンバーをプールに追加する。** 現在のプールは全て公開OOF由来で、**互いに相関が高い**。ResNet+FE系やNN Residual系のような、異なる帰納バイアスを持つ自前モデルを1〜2本足す方が、既存プール内での重み最適化より期待値が高い。
2. **ブレンド重み W をpublic LBで選ぶのをやめ、CVで選んだ値に固定する。** 著者自身がW選択の危険性を実証しているのに、最終的に採用した W=0.35 は「曲線を地図化して」選んだもの。自分の分析に従うなら、OOF上で選んだ重みをそのまま使い、public LBを一切見ないのが一貫している。
3. **自己参照メンバーの検出を、正規表現ではなく統計的に行う。** 現在 `^(naji|sz_naji|v13_anchor|hb_candidate)` という**名前ベース**の除外。命名規則が変われば漏れる。代わりに「そのメンバーのOOFを、残りのプールで回帰したときの決定係数 R²」を計算し、閾値を超えたら自己参照と判定する方が頑健。
4. **Ladderメカニズムの実装。** 提出のたびに「前回ベストを 0.0002（=分解能の3倍程度）以上上回ったときだけ更新する」という規律をコードで強制する。本notebookの洞察を、次のエピソードで自動的に効かせられる。
5. **private相当のシミュレーションを、40試行から1000試行へ拡張して信頼区間を出す。** 現在の40試行では「大多数の試行で悪い」という定性的主張までしかできない。試行数を増やせば「W=0.35 を選ぶことで private が期待値でいくら下がるか」を区間推定できる。

---

## 3. RSNA Knee Abnormality Detection

**`Knee MRI: twelve findings from a single model`** — 原著者: DREAD DEVELOPMENT / Public 0.924 / 実行 53秒 / 102 votes
元notebook: https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model

膝MRIから12所見（ACL断裂、MCL断裂、内外側半月板断裂、内外側・膝蓋大腿OA、関節液貯留、滑膜炎、ベーカー嚢腫、骨挫傷、骨折）を同時判定。**単一モデル・アンサンブルなし・TTAなし**で0.924。

### 評価指標の要約

**macro-AUC**（12所見それぞれのROC AUCの単純平均）。症例数で重み付けしないため、**骨折やベーカー嚢腫のような稀な所見を捨てる戦略に高得点を与えない**。臨床的には稀な所見の見落としこそ問題なので妥当な設計。本notebookはこれをアーキテクチャに直結させ、**所見ごとに独立したattention重み**を持たせている（ACL断裂は矢状断の数枚、OAは冠状断の多数枚に分布するため、プーリングを共有させられない）。macro平均では1所見の改善が全体の1/12を動かすので、この設計は直接スコアに効く。

### 学べる主要テクニック

- **アンサンブルを測って捨てた**: 45症例のゴールドセットでは3本ブレンドが +0.010（0.9025→0.9068）。しかし**実際のLBでは 0.914 → 0.914〜0.915** にしかならず、著者は「ゴールドセットのノイズだった」と結論して1本に戻した。実行時間も1/3に。
- **改善の主張にブートストラップを付けている**: 学習データ拡張（3,155→4,349症例）は 0.8923→0.9054、**2000回ブートストラップの92.7%で優位**。アンサンブルは捨て、こちらは採用——**差の大きさではなく信頼性で判断している**。
- **LLMでレポートをソフトラベル化**: 構造化ラベルは4,407症例中58件のみ。残りは自由記述の読影レポート。LLMで読んで12個の**確率**に変換（「断裂が疑われる」→ 1 ではなく 0.8）。58症例は一切学習に使わず評価専用（macro-AUC 0.9167）。
- **5固定スロット×64スライスの入力設計**: 矢状断18+14、冠状断12+8、軸位断12。一部スロットで fluid-sensitive を優先し、一部で優先しない（水感受性は腫脹・関節液を、非水感受性は解剖・軟骨をよく写し、12所見は両方に分かれる）。欠損スロットはゼロ埋め＋maskでスキップ。
- **mm単位クロップ**: PixelSpacingを使い中心から140mm四方を切ってから336pxへ。0.3mm/pxでも0.5mm/pxでも膝がフレームに占める割合が一定になり、医学的に無意味なスケール差を学習させない。
- **DICOMメタデータの3つの罠**: (a) スライス順は ImageOrientationPatient の法線への ImagePositionPatient の射影で決める（ファイル名順ではない）、(b) `MONOCHROME1` は白黒反転が必要、(c) PixelSpacing。いずれも**エラーにならず静かに精度を下げる**。
- **周辺スライスを捨てない**: 6%〜94%（従来は15%〜85%）。側副靭帯と外側半月板は周辺スライスに存在し、切るとその所見のAUCが落ちてmacro平均を直撃する。**解剖学の知識がそのままハイパーパラメータになっている**。
- **Code Competitionの完遂設計**: fp16失敗時のfp32フォールバック、症例単位の例外捕捉（0.5を入れてログして続行）、アームの逐次実行によるRAM節約、競技DICOMツリーの再帰globの禁止、"column order drift"/"row identity drift" と名付けられたassert。

### 改善点の考察

**他notebookとの比較（同コンペ上位を確認）**
- 上位帯 `RSNA Knee | DINOsaur V4`（0.936）、`RSNA Knee: Take Care Of Your Knee`（0.936）、`rsna-base`（0.936）はいずれも**多モデルのランクアンサンブル**で、本notebookとは 0.012 の差がある。本notebookが採用していないのは (a) DINOv2/DINOv3系の自己教師あり事前学習バックボーン、(b) TTA、(c) 複数バックボーンのブレンド。
- ただし著者はアンサンブルを**測った上で**捨てているので、0.012の差の出所は「アンサンブル」ではなく **バックボーンの選択（DINOv3 vs CoAtNet）** にある可能性が高い。実際 `RSNA Knee DINO-RadImageNet Rank Ensemble`（0.920）は本notebookとほぼ同等で、DINOだけでは足りないことを示唆する。
- `RSNA Knee: read the report, then the knee`（0.906、245 votes）も読影レポートを使う方向性だが、スコアは本notebookが上回る。

**関連文献**
- [Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/) — **まさに同じ12所見・膝MRI**を扱った研究。本notebookが「所見ごとのattention」を持つのに対し、こちらは**シーケンス間（co-plane）のattention**を提案している。本notebookは5スロットを固定順で連結しているだけで、スロット（＝撮影面/シーケンス）間の関係は明示的にモデル化していない。
- [OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI (arXiv 2602.20752)](https://arxiv.org/pdf/2602.20752) — 筋骨格MRIの基盤モデル。8つの膝所見カテゴリで **macro-AUROC 0.908**。本notebookの58症例ゲートでの 0.9167 と近い水準。
- [Multi-task deep learning for nine common knee abnormalities (eClinicalMedicine)](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(25)00467-5/fulltext) — 大規模多施設での段階的検証。臨床応用側から見た評価設計の参考。

**改善提案**
1. **co-plane attention の導入。** 現在は5スロットを固定順で連結し、attentionはスライス方向にのみかかる。文献にある通り「どの撮影面がどの所見に効くか」を明示的にモデル化すれば、macro平均で効く稀な所見（外側半月板、骨折）を伸ばせる可能性がある。既存の `RaptorClassifier.head` に、スロットID埋め込みを足すだけで実装できる。
2. **バックボーンをDINOv3系に置き換えて、同じ入力設計・同じソフトラベルで比較する。** 上位帯との0.012差の出所を切り分けられる。本notebookの前処理とラベル生成は明らかに他より丁寧なので、バックボーンだけ差し替える実験の価値が高い。
3. **ソフトラベルの品質を58症例ゲートで検証する。** LLMがレポートを確率に変換した結果が、実際のラベルとどれだけ一致するか（所見ごとのAUC/Brierスコア）を測っていない。一致が悪い所見があれば、そこがmacro平均のボトルネックになっている可能性が高い。
4. **58症例のゴールドセットが小さすぎる問題に対処する。** 著者は45症例での判断が誤りだったことを自ら認めているが、58症例も十分とは言えない。**ソフトラベル側でのcross-validation**（4,349症例でのOOF macro-AUC）を主指標にし、58症例ゲートは最終確認に留める方が、判断の解像度が上がる。
5. **TTAを再検討する（ただし測ってから）。** 著者はTTAを使っていないが、この設計では **`K_EVAL=42` のウィンドウ選択自体が一種のTTA**になっている。左右反転TTAは膝の解剖学的左右差を考えると危険だが、わずかな平行移動やコントラスト摂動は安全に足せる。**ただし追加するなら、アンサンブルを捨てたときと同じ厳しさで測ること。**

---

## まとめ

本日の3本は、スコアの帯も分野もばらばらだが、**「改善の主張には検証を伴わせる」という一点で完全に一致している**。

- Biohubは「設定した状態」と「実際にロードされた状態」を分けて出力し、静かな失敗を可視化する。
- Playgroundは自分が争っている差が測定分解能以下であることを計算で示し、過去7エピソードの実データで裏付ける。
- RSNAは45症例で見えた +0.010 を「ノイズだった」と認めてアンサンブルを捨て、代わりにブートストラップで92.7%優位を確認できた改善だけを採用する。

**高スコアnotebookを読むとき、「どうやって上げたか」と同じくらい「著者がそのスコアをどれだけ疑っているか」を見るべき**、というのが本日の結論。疑っていない高スコアは、たいていpublicへの過学習である。
