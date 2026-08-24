# 2026-08-25 Kaggle日次レビュー

今日の3本は、偶然にも**「スコアを上げた仕組みは何か」を自分で説明できるかどうか**が
くっきり分かれる並びになりました。1本目は検出アンサンブル＋制約付き最適化という王道、
2本目は公開OOFを205本集めた大規模スタッキング、3本目は**学習を一切しない後処理だけの提出**です。

---

## 1. Biohub - Cell Tracking During Development（固定枠）

- **notebook**: [0.926-biohub-divsub](https://www.kaggle.com/code/rockerritesh/0-926-biohub-divsub)
- **原著者**: ROCKER RITESH
- **Public Score**: 0.926（高いほど良い / GPU T4 x2・22分）
- **ローカル成果物**: `biohub_0-926-biohub-divsub.ipynb`

### 学べる主要テクニック

1. **検出段アンサンブル**: 独立シードの TemporalUNet3D 2本の検出ロジットを 0.525 : 0.475 で融合し、
   点抽出は1回だけ行う。「アンサンブルは上流で、最適化は下流で1回」という役割分担。
2. **8方向フリップTTA**: 上流スクリプトを文字列置換（モンキーパッチ）してTTAのビュー数を拡張。
   細胞のように向きに意味がない対象では確実に効く安価な改善。
3. **非対称なILPコスト**: appearance（トラック開始）罰則 0.0 / disappearance（途中終了）罰則 1.5。
   「視野に入るのは自然、途中で消えるのは不自然」という物理的直観をコストに直接埋め込んでいる。
4. **幾何ゲートによる分裂の抑制**: 親子8.0µm・姉妹11.0µm・t+2で2.25µm以上離れる、という3条件。
   原著は「元パイプラインの12/15/10では自分の点密度では過剰提案（FP 14件→9件）」と**根拠つきで**締めている。
5. **提出物の自己監査セル**: 列・行数・SHA256・グラフのトポロジ不変条件（親≤1・子≤2・t→t+1）を検査してJSON出力。

### 評価指標の要約

CTC系の複合スコアで、**検出の正しさ（DET/TF系）とリンク・系統樹の正しさ（TRA/CT/BC(i)/CCA系）を別々に測って合成**します。
検出だけ・リンクだけでは片方に甘い指標になるためで、1細胞の見逃しが「そのトラック全体の断裂」として二重に効くのが厳しい点。
このnotebookは高い検出閾値（τ=0.96875）で偽陽性を抑え、後段のギャップ修復で見逃しを取り返す二段構えで対応しています。

### 改善点の考察

**他notebookとの比較**（同コンペのCodeタブ上位を確認）:

| notebook | Score | このnotebookが採用していない要素 |
|---|---|---|
| [biohub ct mix divaug](https://www.kaggle.com/code/) | **0.969** | mixup系のデータ拡張 + **division augmentation**（分裂サンプルの明示的な増強） |
| Biohub Solution | 0.966 | （5データセット入力、より大きなアンサンブル） |
| Biohub Competition Solution | 0.965 | 同上 |

**0.926 と 0.965〜0.969 の間には約0.04の壁があります。**上位群は「divaug」＝**分裂イベントの増強**を
名前に掲げており、本notebookが幾何ゲートで分裂を*抑制*する方向に振っているのと対照的です。
分裂は数が少なく（クラス不均衡）、かつ間違えると系統樹全体を壊すため、
「抑制して安全にとる」か「増強して当てにいくか」がスコア帯を分けている可能性が高い。

**関連文献**: [Higher-Order Cell Tracking Transformer (HOCT, arXiv 2607.11754)](https://arxiv.org/abs/2607.11754) は、
**候補リンク（エッジ）同士が互いにattentionを張る edge-centric なTransformer**で、3D幾何事前分布のもとに
高次の整合性を直接学習し、CTCベンチマークでSOTAを達成しています。ILPの寄与についてのablationも含まれており、
「学習でどこまでやり、ILPで何を担保するか」の境界を考える上で直接参考になります。

**改善提案**:

1. **分裂の扱いを抑制から増強へ振り直す**。上位notebookが `divaug` を掲げている以上、
   まず分裂サンプルの拡張（時間反転・回転・合成分裂）を試し、幾何ゲートは**緩めた上でILPコストで制御**する。
2. **シードを2→4に増やす**。現在は 0.525:0.475 のほぼ半々。シード数を増やして検出場を平均すると、
   高閾値 τ=0.96875 を維持したまま見逃しを減らせる可能性がある（コストは推論時間のみ）。
3. **モンキーパッチに検証を入れる**。`_s.replace(_old,_new)` は失敗しても例外を出さない。
   `assert _old in _s` を前に置くだけで「TTAが実は無効だった」という最悪のサイレント失敗を防げる。
4. **エッジ確率の閾値 τ_e とILP重みを同時最適化する**。現状は個別に手で決めた値。
   検出閾値・エッジ閾値・disappearance重みは互いに補償し合う関係なので、3次元のグリッド／Optunaで
   同時に探索したほうが良い局所解に届く。
5. **HOCT的な「エッジ間attention」を後処理に混ぜる**。全面置換でなくとも、
   ILPに入れる前にエッジ候補を高次整合性でリスコアリングするだけで、ILPの探索空間が素直になる。

---

## 2. Playground Series S6E8 — Predicting Smartphone Addiction

- **notebook**: [S6E8 Rank-Logit-Regime Fusion | LB0.97125](https://www.kaggle.com/code/hboyang/s6e8-rank-logit-regime-fusion-lb0-97125)
- **原著者**: BYER (hboyang)
- **Public Score**: 0.97125（本日時点で公開notebook最高 / 残り7日）
- **ローカル成果物**: `playground_s6e8-rank-logit-regime-fusion-lb0-97125.ipynb`

### 学べる主要テクニック

1. **rank と logit の二重表現**: 各メンバー予測を `rank01`（順序情報）と `logit`（確信度情報）の
   両方の列として並べる。AUCは順序しか見ないがrank化は確信度を捨てるため、両方渡して線形モデルに選ばせる。
2. **レジーム特徴量（交互作用）**: 欠損ゼロフラグ・欠損4個以上フラグ・**メンバー間の意見のばらつき**を
   融合特徴量に掛け合わせ、「状況によってメンバーの重みを切り替える」ことを線形モデルのまま実現。
3. **float64 + LBFGS + チャンク勾配**: 1205列×691k行のfloat64行列を、
   **数学的に同一の全バッチ勾配を部分和に分割**することで16GBのT4に収める。精度を落とさずメモリ制約を外す好例。
4. **fold定義の凍結をdocstringで契約化**: `StratifiedKFold(5, shuffle=True, random_state=42)` を
   コミュニティ共通の凍結値として明記し、配列は**位置対応（IDなし）**であることも行数つきで宣言。
5. **順位空間での最終混合**: dual と regime の2ストリームを `rank01` に直してから 0.55:0.45 で混ぜる。
   スケールの違う出力を確率のまま混ぜると混合比が意図どおりに効かない。

### 評価指標の要約

**ROC-AUC**（二値分類、train 691,369行 / test 296,302行）。予測値の絶対値ではなく**順序のみ**を見るため、
全メンバーを最初にrank化してスケール差を消す設計が指標と直結しています。
LB 0.9712帯では上位の差が0.0001程度なので、単体の強さより**誤りの非相関性**が効きます。

### 改善点の考察

**他notebookとの比較**（同コンペのCodeタブ上位を確認）:

| notebook | Score | 対比 |
|---|---|---|
| S6E8 Rank-Logit-Regime Fusion（本日） | **0.97125** | 205メンバー + レジーム交互作用 |
| S6E8: Elite Rank Average Ensemble | 0.97123 | 単純なrank average。**ほぼ同点** |
| S6E8 Addiction LB 0.97113 | 0.97123 | — |
| [Why Every S6E8 Notebook Above 0.97110 Overfits](https://www.kaggle.com/code/) | 0.97115 | 「0.97110超えは全部過学習」という主張 |

**注目すべきは、205メンバー＋交互作用という複雑な機構が、単純な Elite Rank Average（0.97123）に
たった 0.00002 しか勝っていない**ことです。この差はPublic LBのノイズ幅に完全に埋もれています。
つまり本notebookの複雑さは、**現時点では正当化されていない**。

**関連文献**: [How Ensemble Learning Balances Accuracy and Overfitting (arXiv 2512.05469)](https://arxiv.org/pdf/2512.05469) は、
アンサンブルの効果を「制御された分散削減」として定式化し、
**リーダーボード順位ではなく generalization gap（train/valid差）を診断指標として使うべき**だと論じています。
本notebookが `VALIDATE = False` を既定にしてCV評価パスを走らせない設計であることと、まさに対照的です。

**改善提案**:

1. **`VALIDATE = True` で回して、honest CV と LB の乖離を測る**。これをやらない限り、
   0.97125 が実力なのかPublic LBへの適合なのか判別できない。最優先。
2. **メンバーのプロヴェナンス階層化**: 205メンバーのうち、fold凍結を本当に守っているものだけを
   「Tier 1」として分離し、Tier 1だけで組んだ場合のスコアと比較する。
   読み込み時に各メンバー単体のOOF AUCを表示し、公称値と一致しないものを弾く検証を入れる。
3. **`MIX_W=0.55` / `FUSION_C=3.5` / `missing>=4` の由来を検証する**。
   これらがLB調整で決まったなら、CV上で選び直すと値が変わるはず。変わるなら過学習の証拠。
4. **メンバー数を段階的に削る実験**（205→100→50→20）。スコアがほぼ落ちないなら、
   複雑さは不要であり、Private LBでは少数精鋭のほうが安定する可能性が高い。
5. **提出の選択**: 最終提出2枠のうち1つは、**単純なrank averageの堅牢版**を選ぶ。
   同点に近い2案があるなら、shake-downの小さいほうを保険として持つのが合理的。

---

## 3. UMUD Challenge: Muscle Architecture in Ultrasound Data（開催中の実コンペ）

- **notebook**: [[LB 0.76704] No-Train Anatomy-Calibrated DLTrack](https://www.kaggle.com/code/phuongncn/lb-0-76704-no-train-anatomy-calibrated-dltrack)
- **原著者**: phuongncn
- **Public Score**: 0.76704（**低いほど良い** / 学習なし・CPU・25秒 / GPL-3.0）
- **コンペ**: 165チーム・賞金3,000 CHF・残り約3ヶ月・主催 University of Basel
- **ローカル成果物**: `competition_lb-0-76704-no-train-anatomy-calibrated-dltrack.ipynb`

> 15日連続でRSNA Knee枠が続いていたので、今日は同じ医療画像分野の別コンペに切り替えました。

### 学べる主要テクニック

1. **キャリブレーションだけでスコアを動かす**: MTに一律 -0.4mm の系統バイアス補正をかけただけで
   0.77681 → 0.76794。モデルを強くするより費用対効果が高い局面がある、という実例。
2. **ドメイン知識を後処理ゲートとして使う**: `FL·sin(PA) ≥ MT` という羽状筋の幾何関係を、
   損失関数ではなく**ブール値の整合性投票**として使い、補正の適用可否を行ごとに切り替える。
3. **`Decimal(prec=60)` と ROUND_HALF_EVEN**: 出力をビット単位で再現する必要があるとき、
   float の2進表現では不可能。偶数丸めは多数行にわたる丸めバイアスの蓄積も防ぐ。
4. **フェイルクローズド設計**: 入力のSHA-256・列名・行数(309)・ID一意性を全部検証し、
   1つでも外れたら例外。「想定外の入力でそれらしい出力を出す」ことを構造的に禁止している。
5. **base64+gzipによるソース埋め込み配布**: 外部データセットに依存せず、notebook単体で完全再現できる。
6. **否定的結果の記録**: 「広範なFL補正は0.82498（大幅悪化）」「停止規則なしの継続は不採用」まで表に残している。

### 評価指標の要約

**UMUD Score** = PA(度)・FL(mm)・MT(mm) の3変数それぞれのMAEを、変数ごとの**許容値(tolerance)で正規化して合成**したもの。
**小さいほど良い**。単位も典型スケールも違う3変数を対等に扱うための正規化であり、
RMSEでなくMAEなのは画質のばらつきによる少数の大失敗に指標を支配させないためです。
指標がtolerance正規化MAEであるがゆえに、**系統バイアスの除去が最も効率のよい改善手段**になります。

### 改善点の考察

**他notebookとの比較**（同コンペのCodeタブを確認）:

| notebook | Score | 中身 |
|---|---:|---|
| [LB 0.76704] No-Train（本日） | **0.76704** | 学習なし・採点済みCSV + 後処理チェーン |
| UMUD Quick and Dirty | 1.33851 | 幾何ベースラインの簡易実装（31票） |
| Muscle architecture - PA FL MT | 1.87113 | — |
| U-Net Segmentation for Muscle Architecture | 未採点 | **実際にセグメンテーションを学習している唯一の系統** |
| UMUD Challenge: Muscle Architecture in Ultrasound | 5.12207 | 初期ベースライン |

**構図が非常にはっきりしています**: 0.767 の1本だけが飛び抜けていて、次点が 1.34。
そしてその1本は**新しい画像に一切適用できないルックアップテーブル**です。
つまりこのコンペの公開notebook群には、**実際に動くモデルで 1.0 を切ったものがまだ存在しない**。
逆に言えば、まっとうなセグメンテーションパイプラインを組むだけで公開2位に入れる余地があります。

**関連文献**: このコンペの土台になっているのが
[DL_Track_US（Ritsche et al., Ultrasound in Med & Biol 2023）](https://pubmed.ncbi.nlm.nih.gov/38007322/)
です（コンペ主催者の Paul Ritsche 本人の研究）。classic U-Net と VGG16事前学習エンコーダ付きU-Net の2本で
筋束と腱膜をセグメンテーションし、**FLで手動比較 -2.4mm、PAで1.5°以内、MTで1mm未満**の一致を報告しています。
訓練データとコードが公開されているので、**主催者の手法を再現することが最も確実な出発点**です。

**改善提案**:

1. **まずDL_Track_USを素直に再現し、CVでベースラインを引く**。0.767のアンカーは再現不能なので、
   自分の測定パイプラインを持つことが先決。事前学習済み重みは規約上使用可（要明記）。
2. **キャリブレーションは再現可能なパイプラインの上に乗せる**。本notebookが示した
   「MTの系統バイアス除去」「`FL·sin(PA)` 整合ゲート」自体は正しい発想で、**モデル出力に対して適用すれば普遍的に効く**。
3. **5フレーム連続という構造を使う**。テストは動画から5枚連続で切り出されている。
   同一シーケンス内で予測を**中央値集約**するだけで、フレーム単位のノイズが大きく減るはず（時間的安定性の評価にも効く）。
4. **3変数を独立に予測しない**。`MT ≈ FL·sin(PA)` という制約があるので、
   3出力を独立回帰するのではなく、2つを予測して3つ目を導出する、または
   **整合性違反を罰する損失項**を足すマルチタスク設計にする。
5. **上位入賞にはFAIR/オープンソース要件がある**点に注意。private上位3チームは
   OSIライセンス・FAIRチェックリスト・再現実行の3条件を満たさないと失格になる。
   最初からリポジトリ・requirements.txt・READMEを整えて進めるべき。

---

## 今日の一言まとめ

**「スコアが高い」と「手法が良い」は別の軸**、という当たり前のことを3本が別々の角度から突きつけてきました。
Biohubは0.926で堅実だが上位0.969には0.04負けていて、その差は**分裂を抑制するか増強するか**の設計思想の差にありそう。
S6E8の0.97125は、単純なrank average(0.97123)に0.00002しか勝っておらず、複雑さがまだ正当化されていない。
UMUDの0.76704に至っては、そもそも**新しい画像に適用できない**。
今日一番実用的な学びは、**UMUDのように公開notebookが総崩れしている領域では、
主催者自身の論文手法を素直に再現するだけで上位に入れる**——という機会の見つけ方だと思います。

## Sources

- [Higher-Order Cell Tracking Transformer (arXiv 2607.11754)](https://arxiv.org/abs/2607.11754)
- [Cell-TRACTR: A transformer-based model for end-to-end segmentation and tracking of cells (PLOS Comput Biol)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013071)
- [How Ensemble Learning Balances Accuracy and Overfitting (arXiv 2512.05469)](https://arxiv.org/pdf/2512.05469)
- [Fully Automated Analysis of Muscle Architecture from B-Mode Ultrasound Images with DL_Track_US (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38007322/)
- [Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning (arXiv 2009.04790)](https://arxiv.org/pdf/2009.04790)
