# 2026-09-03 Kaggle日次レビュー

本日の共通テーマ：**「効いた」を主張する前に、効いたことを見える形にする**

3本とも公開notebookの実力最上位帯（Biohub 0.935 / S6E9 0.94607 / RSNA 0.935）でありながら、
中身を読むと差がついているのは**モデルではなく実験の管理方法**でした。
Biohubは「変更は1点だけ」と宣言して設定ドリフト検出とパイプライン・マニフェストで裏付け、
RSNAは各段をtry/exceptで囲んで失敗を必ず印刷し、
S6E9は外側fold×内側CVの二重防御でOOFスコアの信用を守る。
一方で3本に共通する弱点も同じで、**積み上げた要素を個別に評価（ablation）していない**ことでした。

---

## 1. Biohub - Cell Tracking During Development（固定枠）

| 項目 | 内容 |
|---|---|
| notebook | [Biohub Agreement-Gated Dual-Seed Fusion](https://www.kaggle.com/code/flexonafft/biohub-agreement-gated-dual-seed-fusion) |
| 原著者 | Igor Zharov (flexonafft) |
| スコア | Public LB **0.935**（V9）/ 27 votes |
| ローカル | [`biohub_biohub-agreement-gated-dual-seed-fusion.ipynb`](./biohub_biohub-agreement-gated-dual-seed-fusion.ipynb) |

### 学べる主要テクニック

- **Agreement gate（合意ゲート）**：2つの乱数シードで学習した検出器の確率を**調和平均的に融合**（`harmonic_probability`, 重み0.15）。算術平均と違い片方が低いと一気に下がるため、「両方が同意した検出だけを強い候補にする」フィルタとして働く。
- **ILP（整数線形計画）による軌跡構築**：出現コスト0.0／消失コスト2／分裂重み1.2を与え、SCIPで全体最適なエッジ集合を解く。「軌跡を途中で終わらせるのは高くつく」という指標の構造をそのままコストに翻訳している。
- **DeepCenter を veto 専用に使う**：別途学習した3D U-Netを予測器としてではなく**拒否権**として使い、分裂を追加してよいかを閾値で判定。今回の変更点はこの閾値 0.12 → **0.14** のただ1点。
- **設定ドリフト検出（configuration guard）**：期待する設定値をハードコードし、ズレたら実行前に例外。fail fast。
- **パイプライン・マニフェスト**：「設定した値」ではなく**実際に有効になった状態**（重みファイルが実在したか、DeepCenterが本当にロードされたか）を1か所に印刷する。原著いわく、これまで見つかった静かなバグは全部この印刷ひとつで見えたはずだった。
- **公式指標のオフライン再実装**：`linear_sum_assignment`（ハンガリアン法）による7µm制約つき二部マッチングを自分で書き、LB提出枠を使わずに評価する。

### 評価指標の要約

提出物は `node` 行（t, z, y, x）と `edge` 行（source_id → target_id）を縦積みした**グラフ**。指標は ①同一フレーム内で7µm以内という制約下の最小コスト二部マッチングでノードを対応付け、②リンクの正誤、③分裂の正誤（主催者の修正後仕様：単一連結成分＋両娘系統のカバー＋アンカー＋フォーク）を評価する複合スコア。1フレームの取りこぼしが以降の全フレームに伝播するため「点の当たり外れ」ではなくグラフ一致度を測る必要があり、極端に少数だが価値の高い分裂を独立項として重み付け、水増し防止にノード数ペナルティを課している。

### 改善点の考察

**他notebookとの比較**

| 比較対象 | 採用しているが本notebookに無いもの |
|---|---|
| [biohub-0-948-reproduction-20260901](https://www.kaggle.com/code/cloudssdut/biohub-0-948-reproduction-20260901)（0.935） | **SimpleNodeTransformer による学習済みエッジスコアラー**。本notebookは距離＋関連付けスコアのヒューリスティックなコストだが、こちらはエッジのスコアリング自体をTransformerに学習させている |
| [biohub-948_sew20](https://www.kaggle.com/code/rishabhr0y/biohub-948-sew20)（0.935） | `secondary_edge_weight` の単一ノブ掃引を明示的に記録。同じ「1点だけ変える」でも**掃引の全結果を残している** |
| [Biohub Metric Hack](https://www.kaggle.com/code/anvithpothula/biohub-metric-hack)（0.95）系 | 指標の抜け穴を突く手法。**採用すべきではない**が、LB上位帯の 0.95〜0.966 がこれらであると知っておくことは、自分の 0.935 を正しく位置づけるうえで重要（honest な公開上限は約0.935、LB全体の1位は0.963） |

**関連文献**

- [Higher-Order Cell Tracking Transformer (arXiv:2607.11754, 2026-07)](https://arxiv.org/abs/2607.11754) — 著者に本コンペ主催の Royer ラボの Loïc Royer が含まれる。**エッジ中心のアーキテクチャ**で、候補リンク同士を3D幾何事前分布のもとで相互にアテンションさせる。論文の指摘が鋭い：「分裂はノード埋め込み空間で系統を絡ませる」「エッジを共有するノード同士のラベル一致はほぼランダムなので、候補グラフのトポロジーはGNNにとって情報を持たない」。深い事前学習画像エンコーダなしでCell Tracking ChallengeのSOTAを達成。**本notebookのILPコスト設計が手作業で近似しているものを、学習で置き換える方向**。
- [Trackastra (arXiv:2405.15700)](https://arxiv.org/html/2405.15700v1) — Transformerベースの細胞追跡。コンペの一部notebookが既に利用している。

**改善提案（優先度順）**

1. **積み上げた機構を個別にablationする（最優先）**。現在のnotebookは gap closing・short track rescue・dual-seed融合・DeepCenter veto・bidirectional融合の5つが同時に有効。せっかく公式指標のオフライン検証器を持っているのだから、各機構をOFFにした5回の実行で寄与を数値化すべき。「1回の実験で1つだけ変える」という宣言が、**積み上げ済みの部分には適用されていない**のが最大の矛盾。
2. **今回の変更（0.12 → 0.14）の効果を検証器で測ってから提出する**。閾値を1点だけ動かした結果、分裂の追加数が何個減り、指標の分裂項が何点動いたのかを cell 13 の診断で可視化できるはず。LBの1回の返答（0.935）だけでは、変更が効いたのか誤差なのか判断できない。閾値を 0.10 / 0.12 / 0.14 / 0.16 / 0.18 と掃引して**曲線として見る**べき。
3. **エッジスコアリングを学習ベースに置き換える**。上記のHOCTやSimpleNodeTransformerの方向。現在のILPコストは距離と検出確率の手組みで、`GAP_CLOSE_UM=5.8` のような定数が20個以上並んでいる。これらは本来データから学習できる量であり、手動チューニングの上限に達している可能性が高い。
4. **dual-seed を tri-seed 以上に拡張し、合意ゲートの閾値を可変にする**。現在2シードの調和融合だが、3シード以上なら「3つ中2つが同意」といった多数決ゲートが設計でき、`MIN_CANDIDATE_RETENTION=0.90` のような固定の安全弁が不要になる。計算コストとのトレードオフはあるが、推論時間は33分と枠に余裕がある。
5. **救済（rescue）の上限がバインドしているかを印刷する**。`MAX_NODES_FRAC=0.012` / `MAX_NODES_ABS=120` の上限に**実際に張り付いているのか**が分からない。張り付いているなら上限が性能を制限しており、余裕があるなら条件（平均エッジ確率0.88）のほうが効いている。マニフェストに1行足すだけで分かる。

---

## 2. Playground Series S6E9 — Predicting Electric Vehicle Purchases

| 項目 | 内容 |
|---|---|
| notebook | [S6E9 LightGBM](https://www.kaggle.com/code/kirill0212/s6e9-lightgbm) |
| 原著者 | cstdy (kirill0212) |
| スコア | Public LB **0.94607** / 34 votes / Silver（公開notebook最上位帯） |
| ローカル | [`playground_s6e9-lightgbm.ipynb`](./playground_s6e9-lightgbm.ipynb) |

### 学べる主要テクニック

- **桁分解特徴量（digit features）**：`(x // 10**k) % 10` を k = −4..3 で作り、13列 → +104列。合成データの生成器が残す「丸め幅・格子の指紋」を、木モデルが1回の分割で使える形にする。木は周期的・非単調な構造の表現が極めて苦手なので、明示的に与える価値が大きい。
- **頻度エンコーディング**：train+testを結合して各値の出現割合を計算。**yを使わない変換なので結合してもリークにならない**（transductive）。この境界線の理解が重要。
- **入れ子ターゲットエンコーディング**：外側5-foldの学習側だけで `fit` し、さらに `catstat.TargetEncoder(cv=5)` で内側交差適合。二重の防御でOOFスコアの信用を守る。加えて `smooth ∈ {10, 'auto'}` の**両方を特徴量として並べ**、最適値の選択をモデルに委ねる。
- **列の機械的な掃除**：相関行列の**上三角だけ**を見て（`np.triu` で下半分をNaN化。しないとA-B, B-Aで両方消える）相関1.0の重複列を片方落とす＋定数列を落とす。
- **モデル・スイッチボード**：`model_type` 1つで lgb / xgb / cat / TabM / RealMLP / MASA-FT-Transformer を差し替え、fold分割を固定して比較可能に。OOFとテスト予測を `{model_type}_{SEED}_{N_FOLDS}` 命名で `.npy` 保存し、後段のブレンド最適化に備える。
- **mean AUC と pooled OOF AUC の両方を印刷**：この2つの乖離は fold 間で予測スケールが揃っていないサインで、ブレンド前の必須チェック。

### 評価指標の要約

**ROC-AUC**。正例スコア＞負例スコアとなるペアの割合で、しきい値に依存せずクラス不均衡にも安定。決定的な性質は**順位不変**であること——予測値を単調変換してもAUCは1ビットも変わらないので、**確率のキャリブレーション単体では原理的に1点も上がらない**。効くのは順位を実際に入れ替える操作、すなわち新特徴量・別アーキテクチャ・モデル間ブレンドのみ。逆にランク平均アンサンブルはAUCと極めて相性が良い（スケールを気にせず混ぜられる）。

### 改善点の考察

**他notebookとの比較**

| 比較対象 | 採用しているが本notebookに無いもの |
|---|---|
| [S6E9 starter: how to tell a real gain from noise](https://www.kaggle.com/code/georgymamarin/s6e9-starter-how-to-tell-a-real-gain-from-noise) | **意味のない列を足したときスコアがどれだけ動くかを実測し、「改善」の下限（ノイズフロア）を確定する**という手続き。さらに「4つのアイデアを試して、報われた1つ」「この計測器を騙す唯一の方法」「LBに分かること・分からないこと」と、判断の枠組みそのものを提供している。本notebookに最も欠けている視点 |
| [S6E9 Smart Weighted Rank Ensemble](https://www.kaggle.com/code/amanatar/s6e9-smart-weighted-rank-ensemble)（0.94621） | 複数モデルの**重み付きランクブレンド**。本notebookはOOFを保存しているのにブレンドまで行っていない |
| [S6E9 \| 94.59+ \| Transformer and GBDT Ensemble](https://www.kaggle.com/code/lamhuy8904/s6e9-94-59-transformer-and-gbdt-ensemble)（0.94597） | GBDTとTransformerの実際のブレンド。本notebookは器を用意しただけで、単体（lgb）の結果しか出していない |

**関連文献**

- [TabArena: A Living Benchmark for Machine Learning on Tabular Data (arXiv:2506.16791)](https://arxiv.org/html/2506.16791v1) — 51データセット・16モデル・2,500万実行の継続更新ベンチマーク。結論が本notebookの設計と直結する：**適切にチューニング・アンサンブルすれば TabM や RealMLP は GBDT と同等以上**であり、さらに **「GBDTに単体で勝つモデルを探すより、アンサンブルの中でよく効くモデルを探すほうが重要」**。本notebookが TabM / RealMLP / FT-Transformer の器を用意しているのは正しい方向で、あとは実際に混ぜるだけ。
- [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling (arXiv:2410.24210)](https://arxiv.org/html/2410.24210) — cell 8 の `tabm_k: 32` の中身。1つのMLPに k 個の並行ヘッドを持たせ、重みの大部分を共有したまま k モデル分のアンサンブル効果を1回の学習コストで得る（BatchEnsemble系）。

**改善提案（優先度順）**

1. **器を用意しただけで終わらせず、実際にブレンドする（最優先）**。OOFとテスト予測を `.npy` で保存し、fold分割まで固定してあるのに、単体 lgb の 0.94607 で止まっている。TabArenaの結論通り GBDT×NN のブレンドは最も確実な伸びしろで、上位2本（0.94621 / 0.94597）はいずれもブレンド。`model_type` を4回回して OOF AUC で重みを最適化するだけで、追加の発想は要らない。
2. **ノイズフロアを測る**。上記 starter notebook の手法——ランダムな無意味列を足したときのAUC変動幅を実測する——を先に走らせるべき。S6E9の上位帯は 0.94607 / 0.94621 / 0.94597 と**幅0.0003に3本がひしめいている**。この差がノイズフロアを超えているのかどうかが分からないまま順位を追うのは、計測器の分解能を知らずに測定しているのと同じ。
3. **桁分解特徴量の寄与をablationする**。104列を足しているが、`USE_DIGIT = False` での比較が示されていない。効いていないなら列数だけ増やして `colsample_bytree` を 0.3 まで下げる必要も無かったことになる。k の範囲（−4..3）も、実際に有効な桁だけに絞れる可能性が高い。
4. **順位不変性を利用した後処理の整理**。AUCなので提出値のキャリブレーションは不要。逆に、fold平均を**生の確率ではなくランクで**取れば pooled OOF AUC と mean AUC の乖離が減る。1行の変更で試せる。
5. **`smooth` の掃引をターゲットエンコーディング以外にも広げる**。現在 `smooth ∈ {10, 'auto'}` の2種だが、`seed` も `[42]` の1つだけ。複数seedのターゲットエンコーディングを並べるのは安価な多様性の作り方で、GBDTは不要な列を無視できるのでリスクも小さい。

---

## 3. RSNA Knee Abnormality Detection（開催中の実コンペ）

| 項目 | 内容 |
|---|---|
| notebook | [RSNA Knee: Full 4-arm ensemble v55](https://www.kaggle.com/code/nishantkharga/rsna-knee-full-4-arm-ensemble-v55) |
| 原著者 | Nishant Kharga |
| スコア | Public LB **0.935**（V13）/ 73 votes |
| ローカル | [`competition_rsna-knee-full-4-arm-ensemble-v55.ipynb`](./competition_rsna-knee-full-4-arm-ensemble-v55.ipynb) |

### 学べる主要テクニック

- **順位空間での融合**：すべての混合を `.rank(method='average', pct=True)` のパーセンタイル順位で行う。AUCが順位しか見ない以上、生確率のスケール差は無意味なノイズ。**指標の性質に融合方法を合わせた**教科書的な例。
- **所見別の融合重み**：`_a5_target_w` に12個の重みを個別に持つ（ACL/MCL 0.54、Baker's/Fracture 0.38）。macro AUC＝12個の独立問題という構造を、そのまま実装に落としている。
- **`argsort(0).argsort(0)` による順位化**：1回目で「順位i番目は元のどこか」、2回目で「元の要素は何番目か」。fold平均を取る前に順位化することで各foldが対等に意見を出す。
- **レポートからのLLMソフトラベル**：構造化ラベルは4,407検査中**58件のみ**。残りは自由記述レポートしかない。そこでレポートを言語モデルに読ませ、12個の**確率**に変換（「断裂が疑われる」→ 1ではなく **0.8**）。これで4,349件が学習可能になった。
- **ホールドアウトの規律**：本物のラベル58件を**一度も学習に使わず**、正直な検証に使う（macro AUC 0.9167）。ラベルが希少なときほど守りにくく、しかし守らなければ実力を測る物差しを失う。
- **多段fail-safe**：各armを try/except で囲み、失敗時は①例外の型・メッセージ・トレースバックを全部印刷、②バックアップから既知の良い状態へ復元、③`finally` で後片付け保証、④最終条件（submission.csv の存在）を検査して無ければ明示的に落とす。**静かに壊れることだけは許さない**設計。
- **重みファイルのSHA-256検証**：添付データセットが黙って差し替わる事故を検出。
- **物理単位（mm）での切り出し**：`CROP_MM = 130.0`。DICOMの `PixelSpacing` を使い、装置が違っても常に130mm四方を切る。医用画像では必須。

### 評価指標の要約

**macro-averaged AUC**（12所見のAUCの単純平均）。これは**12個の完全に独立した二値分類問題を平等に平均する**という意味であり、3つの帰結を直接生む：①所見ごとに独立して最適化すべき（全所見に同じ後処理を掛ける理由がない）、②稀な所見と頻繁な所見が等しく1/12を占めるので**データ量に比例した労力配分は誤り**、③順位不変なのでキャリブレーション単体では上がらない。micro平均にすると有病率の高い変形性関節症で総合点が決まり、見逃すと重大な骨折の性能が埋もれる——マクロ平均は「どの所見も等しく大事」という臨床的価値判断の数式化である。

### 改善点の考察

**他notebookとの比較**

| 比較対象 | 採用しているが本notebookに無いもの |
|---|---|
| [RSNA Knee: Take Care Of Your Knee](https://www.kaggle.com/code/anhadmahajan06/rsna-knee-take-care-of-your-knee)（**0.936**） | 本notebookが「組み立て直した」元レシピ。決定的な差は**各段のスコアを明記している**こと：DINOv2 0.899 → A5追加 0.910 → RadImageNet+V18 0.920 → CoAtNet+DINOsaur 0.936。つまり**arm単位のablationが最初から提示されている**。しかもスコアは元のほうが 0.001 高い |
| [RSNA Knee \| DINOsaur V4](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4)（0.936） | 同じ0.936帯。CoAtNet/DINOsaur系の本家 |
| [RSNA Knee Abnormalities - Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)（Kaggle公式・310 votes・21,600 views） | **Efficiency Prize トラック**の公式notebook。ランタイム25秒で毎日更新される。本notebookは4本のバックボーンを直列に走らせる構成で、精度トラックでは通用しても効率トラックでは全く戦えない。**同じ資産で別トラックを狙える**という視点が欠けている |

**関連文献**

- [Enhancing disease detection in radiology reports through fine-tuning lightweight LLM on weak labels (arXiv:2409.16563)](https://arxiv.org/abs/2409.16563) — 本notebookのラベル生成手法そのものの学術版。GPT-4等の大型モデルで疑似ラベルを作り、軽量LLMをinstruction tuningで専門化する二段構え。CheX-GPTがGPT-4をゼロショットラベラーとして5万件の胸部X線レポートに適用し、BERTベースモデルを学習させた事例が示されている。
- [PromptRad: Knowledge-Enhanced Multi-Label Prompt-Tuning for Low-Resource Radiology Report Labeling (arXiv:2605.20052)](https://arxiv.org/pdf/2605.20052) — まさに「低リソース × 多ラベル × レポートラベリング」。本コンペの構造（58件の正解ラベル、4,349件のレポート）に最も近い設定。
- 参考：[Multi-task weak supervision for anatomically-resolved abnormality detection in FDG-PET/CT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7994797/) — レポートから領域単位の不完全ラベルを抽出し、注意機構つきマルチタスクCNNを学習する枠組み。

**改善提案（優先度順）**

1. **arm単位のablationを自分で取り直す（最優先）**。元レシピ（0.936）は各段のスコアを公開しているのに、v55（0.935）は**再組み立ての結果として0.001落ちている**。4本を1ファイルに縦積みしてグローバル空間を退避・復元する構造は、どこかで元と違う状態になっている可能性が高い。まず各armを単独で走らせて 0.899 / 0.910 / 0.920 を再現できるか確認すべき。**0.001の劣化の原因を特定することが、次の0.001を得るより価値がある**。
2. **`_a5_target_w` の12個の重みの導出過程を残す（またはやり直す）**。notebook内に根拠が一切書かれていない。LBフィードバックで手調整したのであれば、**public LBへの過学習リスクが高い**（12個の自由度をLBで最適化するのは、LBを検証セットとして使うのと同じ）。58件のホールドアウトは12所見ぶんの重み最適化には小さすぎるので、レポート由来ソフトラベルでのOOFを重み探索に使うのが現実的。
3. **Efficiency Prize トラックを既存資産で狙う**。4本のうち最も費用対効果の高い1本（おそらくDINOv2単体 0.899、または CoAtNet 単体 0.924）だけを残せば、精度をほとんど落とさずランタイムを数分の1にできる。docstring が「アンサンブルなし・TTAなしで0.924」と明記しているCoAtNet単体は、効率トラックの有力候補。**新しい学習を一切せずに賞金トラックを1つ増やせる**。
4. **アーキテクチャを分離する**。`_A5_SAVED = dict(globals())` でグローバル空間を退避・復元し、`_a5_` / `_rad_` / `_blend_` でプレフィックス衝突を回避する構造は、変更のたびに事故を招く。各armは独立notebookとして走らせ、**submission CSVだけを受け渡す**設計にすれば、arm単位の差し替えとablationが1コマンドになる。原著の設計思想（順位融合・所見別重み・fail-safe）は優れているので、実装の整理だけで再現性が大きく上がる。
5. **稀な所見に労力を集中する**。macro平均なので Fracture や Baker's の1所見が全体の8.3%を占める。これらは症例数が少なくAUCも不安定なはずで、専用のデータ拡張・クラス重み・所見特化ヘッドの投資対効果が最も高い。現状は融合重みを 0.38 に下げているだけで、**「弱いから既存モデルを信じる」という受け身の対処**に留まっている。

---

## まとめ

3本とも公開notebookの実力上限に張り付いており、そこから先で効くのは新しいモデルではなく **「いま積み上がっているもののうち、どれが本当に効いているか」を測ること** でした。

- Biohubは公式指標のオフライン検証器を**持っているのに**、5つの機構を同時に有効化したまま1点だけ動かしている
- S6E9はOOFとfold分割を**固定してあるのに**、単体モデルで止まってブレンドしていない
- RSNAは元レシピが各段のスコアを**公開しているのに**、再組み立てで0.001落ちた原因を調べていない

いずれも「次にやるべきこと」が新規開発ではなく、**すでに手元にある道具を使い切ること**である点で共通しています。
そして S6E9 の starter notebook が言う通り、その前にまず**計測器の分解能（ノイズフロア）を知る**必要があります。
上位3本が幅0.0003にひしめく状況で、その差が実力なのか偶然なのかを判定できなければ、
どの改善を残すかという判断そのものが成立しません。

---

### 出典

- Kaggle: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) / [Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) / [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- arXiv: [2607.11754](https://arxiv.org/abs/2607.11754), [2405.15700](https://arxiv.org/html/2405.15700v1), [2506.16791](https://arxiv.org/html/2506.16791v1), [2410.24210](https://arxiv.org/html/2410.24210), [2409.16563](https://arxiv.org/abs/2409.16563), [2605.20052](https://arxiv.org/pdf/2605.20052)

各notebookの著作権は原著者に帰属します。本フォルダのipynbは学習目的の解説付き写しです。
