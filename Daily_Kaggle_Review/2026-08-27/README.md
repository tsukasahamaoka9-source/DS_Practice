# 2026-08-27 Kaggle日次レビュー

今日の3本を貫くテーマは、**「混ぜる」という同じ動詞の、質の階層**です。

- **Biohub** は、2つのモデルのロジットを**統計量を揃えてから、行ごとに配合比を変えて**混ぜる。
- **RSNA** は、4つのアームを**所見ごとに重みを変え、実行時の相関で自動減衰させて**混ぜる。そして
  「45症例の検証で見えた +0.010 は**ノイズだった**」と気づいてアンサンブルを**やめる**判断まで含んでいる。
- **Playground S6E8** は、2本を固定比で混ぜるだけ——だが**混ぜた結果が元から何ミリ動いたかを自分で測っている**。

3本とも「混ぜた」。違うのは、**混ぜたことを検算しているかどうか**です。
S6E8の公開LB上位には、他人のsubmission.csvを3本平均しただけのnotebookが並んでいます（後述）。
同じ「混ぜる」でも、そこには天と地の差があります。

---

## 1. Biohub - Cell Tracking During Development（固定枠）

- **notebook**: [Biohub Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion)
- **原著者**: IGOR ZHAROV (flexonafft)
- **スコア**: Public 0.926 / Best 0.926（V16）・GPU T4 x2・22分1秒・36 votes
- **ローカル成果物**: `biohub_biohub-harmonic-fusion.ipynb`

### 学べる主要テクニック

1. **モーメントマッチングによるロジット融合**。2つのモデルのロジットは**スケールも中心も違う**ので、
   素朴に足すと出力レンジの大きいほうが支配する。混ぜる前に
   `(secondary − mean) × (primary.std / secondary.std) + primary.mean` で平均と標準偏差を主モデルに揃える。
   スケール比は `clamp(0.5, 2.0)` で挟み、片方が異常に平坦／尖っていても暴走しない。
   **キャリブレーションの違うモデルを混ぜるときの基本作法**で、今日のRSNAでも同じ問題意識が出てきます。
2. **マージン適応ブレンド（タイトルの "Fusion" の実体）**。混合重みを行ごとに変える:
   `clamp(base + secondary_margin − primary_margin, 0.15, 0.75)`。
   マージン＝softmax上位2つの確率差＝確信度の代理。つまり
   **「主モデルが迷っていて、2番手が自信を持っている場所ほど、2番手を重く採る」**。
   固定重みのアンサンブルが全行に同じ配合比を強いるのに対し、これは行ごとに配合を変える。
   さらに両者のトップ候補が一致するときは重みを下げないガードも入っている。
3. **8視点 D4 対称TTA**。元の「反転3種＋原画像＝4視点」に、90°/270°回転・転置・反転転置を足して8視点。
   正方形の対称群 D4 の全要素に相当。向きに意味のない対象（細胞）では予測の向き依存ブレを打ち消せる。
   コストは推論時間で、22分の主因。
4. **密度適応ギャップクローズ**。`GAP_CLOSE_UM = 6.5` を固定せず、周囲3近傍の間隔に応じて
   `GAP_DENSITY_GAIN = 0.040` で自動調整（1ステップ上限 0.125µm）。
   密集域では厳しく、まばらな域では緩く。**「1つの閾値で全領域を裁くのは無理」への実装的な回答**。
5. **分裂検出の3つの構造制約**（コメントが率直で秀逸）。元のルールは「近い孤児同士を結ぶ」だけで
   **FPを数百出しながらTPはゼロ**だった。そこに (C1) 親はトラック途中であること、
   (C2) 姉妹は相互最近傍であること、(C3) **t+2 で両娘が存続し互いに離れていく（DIVERGENCE）**、
   を追加。結果 div_j がコンペ初の非ゼロ 0.0625（TP 1 / FP 9 / FN 6）、総合 0.8799 → 0.8845。
   **「分裂直後の姉妹は離れる」という生物学的事実を、そのまま判別条件にした**のが要点です。
6. **提出ファイルの独立監査セル**。列名・ID連番・row_type集合・データセット名を提出前に検算し、
   違えば例外で落とす。fail fast の実践。

### 評価指標の要約

**Edge Jaccard（重み ≈ 0.85）＋ Division Jaccard（≈ 0.15）の重み付き和**。
Jaccard は `|A∩B| / |A∪B|` で、**取りこぼし（FN）と出しすぎ（FP）を対称に罰する**。
だから戦略は「厳しく検出（閾値 0.96875 + DeepCenterの拒否権）して FP を抑え、
後処理（モーション再リンク・ギャップクローズ）で FN を救済する」という順序になる。
分裂は総数15%しか占めないが、1つ誤ると系統樹の枝が丸ごと壊れて Edge 側に連鎖するため、
`FRAME_FRAC_CAP = 0.0076` / `GLOBAL_FRAC_CAP = 0.00375` という**上限率キャップ**で厳しく抑えている。

### 改善点の考察

**他notebookとの比較**（同コンペ Codeタブ Best Score 上位）:

| notebook | Score | このnotebookが採用していない要素 |
|---|---|---|
| [Biohub Solution](https://www.kaggle.com/code/kaiwalyaraut/biohub-solution) / [Metric_hack_last_call_update](https://www.kaggle.com/code/kaiwalyaraut/metric-hack-last-call-update) | 0.966 | 指標の定義そのものを突く提出（いわゆる metric hack）。学習手法としては別ジャンル |
| [Biohub - Track Your Cells Development](https://www.kaggle.com/code/pilkwangkim/biohub-track-your-cells-development) | 0.927 | 前処理側のブレンド（複数の正規化を混ぜる）。本notebookは単一前処理 |
| [Biohub 0.927 LB](https://www.kaggle.com/code/evgendvorkin/biohub-0-927-lb) | 0.927 | **調和平均による双方向（t→t+1 と t+1→t）融合**。本notebookは順方向のみ |
| [0.926-biohub-divsub](https://www.kaggle.com/code/pilkwangkim/0-926-biohub-divsub) | 0.926 | 分裂に特化したサブモデル |
| **本notebook** | **0.926** | — |

「正攻法」帯の最高値は 0.927 で、本notebook 0.926 はその 0.001 下。
**足りていないのは主に「双方向性」**です。08-26に扱った 0.927 LB は
リンクを順方向と逆方向の両方で評価し**調和平均**（小さい方に強く引きずられる＝実質AND）で統合していました。
本notebookのマージン適応は「2モデル間の対立」を解いていますが、「時間方向の対立」は扱っていません。

**関連文献**:
- [Higher-Order Cell Tracking Transformer (HOCT), arXiv 2607.11754](https://arxiv.org/html/2607.11754) —
  まさにこの弱点を突いた最新研究。「**エッジ中心（edge-centric）**」の設計で、
  ①分裂が系統樹の異なる経路をノード埋め込み空間で絡ませてしまう問題、
  ②**ノードを共有するエッジ同士のラベル一致がほぼランダムになる**問題、の2つを構造的障害として明示的に扱う。
  本notebookが後処理の if 文の束（相互最近傍・発散テスト・キャップ）で人手でやっていることを、
  高次の関係としてモデル内で学習させる方向です。
- [Cell-TRACTR (PLOS Comput Biol, 2025)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013071) —
  track query（過去の履歴を持つクエリ）でセグメンテーションと追跡を end-to-end 化。
  「検出→リンク→後処理」の3段構えを1段にする方向。

**改善提案**:

1. **双方向化して調和平均で統合する**。現状の順方向のみのリンク確率に、逆方向（t+1→t）のリンク確率を
   重み 0.3 程度で調和平均する。0.927 帯との差はほぼここにある可能性が高く、
   実装コストも「同じ推論をもう1回、時間軸を反転して回す」だけで済む。**最優先**。
2. **マージン適応を分裂判定にも広げる**。現在マージン適応が効くのは通常リンクだけで、
   分裂の採否は幾何ゲート（8.0/11.0/10.0µm）と発散テストという**ハードな閾値**に委ねられている。
   「両モデルが分裂に同意したときだけ採用する」という合意条件を足せば、
   FP 9 をさらに削って div_j を伸ばせる余地がある（15%枠なので効果は限定的だが、TP 1 → 2 でも比率としては倍）。
3. **TTAを8視点から絞る**。D4 8視点で22分。実際には z 軸方向の異方性（1.625 vs 0.40625 µm）があるため、
   xy平面内の8視点すべてが等価に有用とは限らない。視点別に寄与を測って**4視点に落とせば実行時間が半減**し、
   浮いた時間を提案1（双方向推論）に回せる。時間はCode Competitionでは有限の資源です。
4. **相互最近傍の「片思い」を捨てずに使う**。C2で相互最近傍でないペアを全部落としているが、
   片思いペアの中にも DeepCenter が強く支持するものはあるはず。
   拒否権（veto）を持つ DeepCenter に**推薦権**も与え、「片思いだが DeepCenter スコアが高い」ものを
   別枠で少数だけ拾う二段構えにできる。
5. **密度適応をギャップクローズ以外にも展開する**。密度で閾値を変える発想が有効なら、
   検出閾値 `DET_THRESHOLD = 0.96875` や分裂半径にも同じ論理が適用できるはず。
   現状これらは全領域で固定値です。

---

## 2. Playground Series S6E8 — Predicting Smartphone Addiction

- **notebook**: [S6E8 Regime-Calibrated Rank Fusion | LB 0.97127](https://www.kaggle.com/code/atakanaldemir/s6e8-regime-calibrated-rank-fusion-lb-0-97127)
- **原著者**: ATAKAN ALDEMIR (atakanaldemir)
- **スコア**: Public 0.97127 / Best 0.97127（V1）・33秒・26 votes
- **ローカル成果物**: `playground_s6e8-regime-calibrated-rank-fusion-lb-0-97127.ipynb`

### 学べる主要テクニック

1. **入力の同一性検査を先にやる**。`列名 == ['id','addicted_label']` / `行数一致` / `id列の完全一致` /
   `全値が有限`。ブレンドで最も多い事故は「**行の並びがずれている**」で、しかもこれは
   例外を出さず静かにスコアだけ壊す。`frame["id"].equals(sample["id"])` の1行がそれを防ぐ。
2. **`locate_unique()`——曖昧さを実行時に持ち込まない**。同名ファイルが2つ見つかったら
   黙って先頭を使うのではなく `FileNotFoundError` で落とす。
3. **タイを作らないパーセンタイル順位**。`(rankdata(x, "average") − 0.5) / n` で開区間 (0,1) に収める
   （後段で logit を掛ける人が無限大で困らない）。そして重みを 0.725 ではなく **0.725001** にする。
   0.725 = 29/40 という有理数だと**順位和が同点になる行が発生**し、
   AUC上で同点は「0.5点（半分正解）」扱い＝順序情報の切り捨てになるため。
   `assert np.unique(prediction).size == len(prediction)` でそれを機械的に保証している。
   **指標の定義を精読していないと出てこない工夫**です。
4. **混ぜた結果の診断**。スピアマン順位相関 ρ、主モデルからの平均絶対順位シフト、95パーセンタイル、最大値。
   アンサンブルが効く条件は「個々が強く、かつ間違い方が違う」こと。
   ρ が 1.0 に近ければ実質同じモデルで、混ぜても公開LBのノイズしか動かない。**混ぜる前に相関を見る**。
5. **出力の SHA-256 を印字**。「versionを上げたのに中身が同じ」「別セルが後から上書きした」を検出できる。

### 評価指標の要約

**ROC-AUC**。「無作為に選んだ陽性1件と陰性1件で、陽性に高いスコアを付けられる確率」。
**順位のみで決まり、単調変換に不変、キャリブレーションを問わない**。
だから確率を平均するより**順位を平均するほうが自然**（片方が自信過剰でも順位に直せば影響が消える）。
本notebookが融合をすべて `percentile_rank` 空間で行い、同点を執拗に排除するのは、この指標構造への直接的な適合です。

### 改善点の考察

**他notebookとの比較**（同コンペ Codeタブ 上位）:

| notebook | Score | 中身 |
|---|---|---|
| [[S6E8] Top 20 Formula: Dual Master Rank Blend](https://www.kaggle.com/code/souvikdbiswas/s6e8-top-20-formula-dual-master-rank-blend) | 0.97128 | 08-26に精読。2マスターのrank blend |
| [S6E8: Elite Rank Average Ensemble](https://www.kaggle.com/code/adarsh1077/s6e8-elite-rank-average-ensemble-0-97092) | 0.97126 | rank平均 |
| **本notebook** | **0.97127** | 2本を 72.5:27.5 で順位融合＋診断 |
| [S6E8 Rank-Logit-Regime Fusion](https://www.kaggle.com/code/hboyang/s6e8-rank-logit-regime-fusion-lb0-97125) | 0.97125 | 本notebookの入力その1（205メンバー融合） |
| [[S6E8] TOP-1 PUBLIC 0.97099](https://www.kaggle.com/code/daniilkrasnovvv/s6e8-top-1-public-0-97099) | 0.97099 | **他人のsubmission.csvを3本読んで rankdata して平均するだけ**（実質4行） |
| [S6E8 Addiction LB 0.97113](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97113) | 0.97123 | 提出ファイルを読んで書き出すだけ。コード内コメントに `# This one is to overfit the LB` |

**この表がこのコンペの現状そのもの**です。上位0.9712帯は 0.00005 以内にひしめいており、
その中身の多くは「他人の提出ファイルの平均」。本notebook 0.97127 と TOP-1 PUBLIC 0.97099 の差は 0.00028。
テスト行数を考えれば、これは**数十行の順序が入れ替わっただけ**の差です。
実際、[Why Every S6E8 Notebook Above 0.97110 Overfits](https://www.kaggle.com/code/szymonkapiski/why-every-s6e8-notebook-above-0-97110-overfits)（0.97115）
という主張のnotebookが64票を集めています。

**本notebookの限界**（原著者も自覚している）: セル4の診断は
**テストデータ上の予測同士を比べているだけで、正解ラベルを使った検証ではありません**。
「2つは違う」ことは示せても「混ぜたほうが良い」ことの証明にはならない。
原著者が "the kind of diversity that **can** help"（助けになり**得る**）と慎重に書いているのは正確な自己認識です。

**関連文献・技術記事**: 順位集約（rank aggregation）でAUCを最適化する際、
重み付き順位平均は Borda count の連続版に相当します。理論的には、
**個々の分類器が独立で同程度の強さのとき**に多数決／平均が最大の利得を生むことが知られており、
逆に相関 ρ が高いほど利得は `1/√(1+(k−1)ρ)` 的に減衰します。
本notebookが測っている ρ は、まさにこの減衰係数を推定する行為です。

**改善提案**:

1. **OOFで検証する（最重要）**。テスト予測同士の相関ではなく、**OOF予測と正解ラベル**で重みを決める。
   S6E8には公開されたOOFライブラリ（[47モデル](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models)、
   [50 weakest OOF](https://www.kaggle.com/datasets/szymonkapiski/s6e8-50-weakest-oof-models)）があるので、
   72.5:27.5 という重みが**OOF上でも最適か**を確かめられる。現状この数字は公開LBへのフィッティングです。
2. **重みの感度分析を出す**。0.70/0.725/0.75 で公開LBがどう動くかを並べる。
   もし 0.70〜0.75 の範囲でLBが 0.00005 しか動かないなら、
   「0.725001 という精度は無意味で、選んだ理由はLBへの過学習」と自分で示せる。
   **感度が低い＝スコア差が実力でない、という証拠**になります。
3. **順位シフトの大きい行を実際に見る**。診断で最大シフトの行を10件抜き出し、
   元データの特徴量を並べる。「2モデルが揉めているのはどういう回答者か」が分かれば、
   ブレンド比ではなく**特徴量側の改善**に手がかりが出る。今は数字を出して終わっている。
4. **ブートストラップでLB差の信頼区間を出す**。今日のRSNA notebookがやっている
   「2000回リサンプリングして92.7%で勝った」と同じことを、公開LBのテスト行に対して行う。
   0.97127 と 0.97125 の差が偶然である確率が示せれば、**この帯での順位争いから降りる**判断ができる。
5. **private split を意識した保険提出を用意する**。残り5日。最終提出2枠のうち1枠は
   このLB最適化版、もう1枠は**OOFで選んだ、LB順位は低いが検証根拠のある版**にする。
   0.9712帯の団子状態は、shake-up（private でのLB大変動）が起きやすい典型的な形です。

---

## 3. RSNA Knee Abnormality Detection（開催中の実コンペ）

- **notebook**: [RSNA Knee | DINOsaur V4 🦖](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4)
- **原著者**: ROMAN TAMRAZOV (romantamrazov)
- **スコア**: Public 0.936 / Best 0.936（V14）・GPU T4 x2・2分37秒・113 votes
- **本日時点で公開notebook中の最高スコア**（2位帯は 0.920〜0.922）
- **ローカル成果物**: `competition_rsna-knee-dinosaur-v4.ipynb`

### 学べる主要テクニック

1. **「アンサンブルをやめる」という判断**。ここが今日いちばんの学びどころ。コード内コメントより:
   > Singles: coatnet384 0.9025 | swinbase384 0.8825 | effv2l480 0.8716.
   > Blend {coatnet+swin+effv2l} = 0.9068 …
   > **on the live leaderboard CoAtNet alone scored 0.914 while every blend scored 0.914-0.915,
   > so ensembling is worth ~+0.001 there — the ~+0.010 it showed on the old 45-study gold set was gold-set noise.**

   **45症例のローカル検証で「+0.010」と見えたものが、実際には +0.001 だった**。
   45症例のAUC差 0.010 は統計的にほぼ誤差。そして3アームなら実行時間3倍。
   「効果が誤差、コストは3倍」なら1本に絞る。
   Kaggleで最も繰り返される失敗——**小さな検証セットのノイズを実力と誤認して複雑さを積む**——への正しい対処の実例です。
2. **代わりに効いたのはデータ量**。学習コーパスを 3,155 → 4,349 studies（+37.8%）に拡張:
   0.8923 → 0.9054（+0.0131、**2000回のブートストラップのうち92.7%で勝利**）。
   「平均が上がった」ではなく「**リサンプリングして何%で勝ったか**」で述べているのが重要。
   しかも伸びたのは Lateral Meniscus +0.071、Fracture +0.057、Lateral OA +0.048 と
   **もともと弱かった所見に集中**。マクロ平均AUCでは弱い所見を伸ばすのが最も効率的。
3. **所見別の融合重み**。`{"ACL": 0.025, "MCL": 0.16, "Synovitis": 0.16, "Fracture": 0.020, ...}`。
   新コーパスで伸びた所見はほぼ純粋に主モデル、伸びなかった所見でだけ旧モデルを 12〜16% 混ぜる。
   **「どこで自分が強いか」を知ったうえで、弱いところにだけ助けを呼ぶ**。
   マクロ平均AUCは所見ごとに独立なので、所見ごとに別々に最適化してよい——指標の分解可能性の直接利用。
4. **相関による重みの自動減衰（両側ガード）**:
   ```python
   if   correlation > 0.992: weight *= 0.50   # 似すぎ → 混ぜても無駄
   elif correlation < 0.65 : weight *= 0.40   # 違いすぎ → 片方が壊れている疑い
   ```
   上側だけでなく**下側にもガードがある**のが良い。相関が異常に低いのは「有用な多様性」ではなく
   「アームの失敗」であることが多い。**静的な設定を、実行時に観測した事実で上書きする**。
5. **全段階が no-op に落ちる防御設計**。2枚目のGPUが無ければlegacyは走らない／
   legacy成功率が98%未満なら丸ごと捨てる／Swinが無ければ完全なno-op／
   study単位で例外が出ても0.5で埋めて続行。コメント曰く "preserves the 0.936 anchor"。
   Code Competitionでは隠れテストでの実行時エラーが即0点なので、この防御は必須。
6. **物理単位でのクロップ（`CROP_MM = 130.0`）**。MRIは機器・患者ごとにピクセルの物理サイズが違う。
   px数で切ると「同じ256pxでも実物大が違う」ことになる。**mmで切ってからリサイズ**するのが医用画像の基本。
7. **fingerprint / SHA-256 による重みの検算**。固定入力を通した出力のハッシュ、
   および重みファイル自体のSHA-256を期待値と照合。
   「重みが読めず初期状態のまま推論していた」という**静かな事故**を検出する。
8. **プーリング設計の多様性**（セル2）。1 study に何十枚もスライスがあり所見が写るのは数枚だけ、
   という構造上、**「どのスライスを重く見るか」がbackbone選択より効く**。
   `LabelAttentionPool` は所見ごとに別の注意重みを学習し、
   `TokenResidualPool` は素朴なプーリングに**ゲート付きの差分だけ足す**（失敗しても素朴版まで劣化するだけ）。

### 評価指標の要約

**12所見の AUC のマクロ平均**。有病率が所見間で大きく違う（関節液貯留は頻繁、骨折は稀）ため、
micro平均だと頻度の高い所見が指標を支配してしまう。マクロ平均は**どの所見にも同じ1票**を与えるので、
「稀だが見落とすと致命的」な所見が軽視されない。
裏返しに、**稀な所見のAUCは分散が大きく、12個の平均という構造がそのノイズを持ち込む**。
本notebookが公開LBとローカルgold setの乖離を繰り返し議論しているのは、この構造ゆえです。

### 改善点の考察

**他notebookとの比較**（同コンペ Codeタブ 上位）:

| notebook | Score | 本notebookとの差 |
|---|---|---|
| **本notebook（DINOsaur V4）** | **0.936** | — |
| [RSNA Knee](https://www.kaggle.com/code/nartaa/rsna-knee) | 0.922 | — |
| [Bend the Knee to DinoV3 (ensembled)](https://www.kaggle.com/code/nartaa/bend-the-knee-to-dinov3-ensembled) | 0.922 | **DINOv3** backbone（本notebookはv2＋CoAtNet） |
| [RSNA Knee DINO-RadImageNet Rank Ensemble](https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble) | 0.920 | — |
| [RSNA Knee frontier v48](https://www.kaggle.com/code/saidmohamedomary/rsna-knee-frontier-v48) | 0.917 | — |
| [RSNA Knee: read the report, then the knee](https://www.kaggle.com/code/pilkwangkim/rsna-knee-read-the-report-then-the-knee) | 0.906 | **放射線科レポート（テキスト）の活用**。本notebookは画像のみ |

0.936 と2位帯 0.922 の差 0.014 は、このコンペでは大きい。
差の主因はコーパス拡張（+0.0131）で、**手法ではなくデータ**です。

**関連文献**:
- [Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities (Nature Communications, 2024)](https://www.nature.com/articles/s41467-024-51888-4) —
  **このコンペとまったく同じ「12所見」設定**の元論文で、本notebookの `co-plane` という用語もここ由来。
  1,748被験者の多シーケンス膝MRIで、シニア放射線科医と同等の性能を報告。
  本notebookは co-plane を「弱い事前情報を注意バイアスとして足す第2ビュー」という**軽い使い方**に留めているが、
  論文は co-plane attention を**学習の主構造**に据えている。伸びしろがある方向です。
- [OrthoFoundation / 多モーダル基盤モデル (arXiv 2601.18250)](https://arxiv.org/abs/2601.18250) —
  膝のX線・MRI **120万枚の無ラベル画像**を DINOv3 backbone で自己教師あり事前学習した筋骨格系の基盤モデル。
  14の下流タスクでSOTA、**MRI構造損傷検出で1位**。ラベル効率が高く、股関節・肩・足首にも汎化。
  本notebookが使う DINOv2（自然画像で事前学習）を**ドメイン内事前学習モデルに差し替える**余地を示しています。

**改善提案**:

1. **backbone をドメイン内事前学習に差し替える**。DINOv2（自然画像 LVD-142M）→ DINOv3 系、
   さらに可能なら OrthoFoundation 系の筋骨格MRI事前学習重みへ。
   2位の「Bend the Knee to DinoV3」がすでに DINOv3 で 0.922 を出しており、
   **本notebookの主アーム（CoAtNet + 拡張コーパス）と DINOv3 の組み合わせは未検証**。
   コーパス拡張が +0.0131 効いたのだから、事前学習ドメインの改善も同程度の効果が見込める。
2. **放射線科レポートを使う**。0.906 の "read the report, then the knee" が示す通り、
   このコンペにはテキスト情報がある。**画像のみのアームとテキスト併用アームは間違い方が根本的に違う**ので、
   多様性という意味では Swin や旧チェックポイントを足すよりはるかに価値が高い可能性がある。
   現在の融合枠（co-plane・legacy・Swin）はすべて「画像を少し違う見方で見る」だけです。
3. **コーパス拡張をさらに進める**。4,349 → 全ラベル済みデータ。
   コメントによれば「以前は 4,349 のうち 3,200 しか使っていなかった」ので、
   まだ未活用のデータや、外部データ（MRNet 等の公開膝MRIデータセット）による事前学習の余地がある。
   **今回の +0.0131 のうち大部分がここから来ている以上、最も期待値が高い方向**。
4. **弱い所見に絞った専用ヘッド／専用しきい値**。伸びた後でも Fracture・Lateral Meniscus は
   相対的に弱い所見のはず。マクロ平均は12等分なので、**最も弱い1〜2所見に計算資源を集中投下する**のが
   スコア効率最大。全所見に同じアーキテクチャを使う必然性はない。
5. **gold set を 58 症例から拡張する**。今回「45症例のgold setのノイズに騙された」と自ら書いている。
   58に増やしたのは前進だが、**58でもAUC差 0.01 の判別は苦しい**。
   ブートストラップの信頼区間を毎回出す運用にするか、
   複数のホールドアウト分割で平均を取る（反復ホールドアウト）ほうが安全です。

---

## 一言まとめ

今日の3本は、はからずも**「アンサンブルをどこまで信じるか」のグラデーション**になりました。

Biohubは、混ぜ方を**行ごとに賢くする**方向へ進んだ。
RSNAは、混ぜても意味がないと**気づいて撤退し**、代わりにデータを増やして +0.0131 を取った。
S6E8は、そもそも混ぜる以外にやることが残っていない飽和状態で、上位が 0.00005 差で団子になっている。

共通して効いていたのは「混ぜる技術」ではなく、**混ぜた結果を測る技術**でした。
モーメントマッチング、相関による自動減衰、ブートストラップでの勝率、順位シフトの分布。
どれも「効いたかどうかをLBスコア1個で判断しない」ための道具です。

そして RSNA のコメント一行——**"the ~+0.010 it showed on the old 45-study gold set was gold-set noise"**
——が、今日の全部を要約しています。小さい検証セットは、いつでも喜ばしい嘘をついてくれる。
