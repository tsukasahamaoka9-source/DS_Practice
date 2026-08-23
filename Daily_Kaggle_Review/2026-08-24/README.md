# 2026-08-24 Kaggle 日次レビュー

本日扱った3本。すべて**学習目的の解説付き写し**であり、原著者のコードは変更していません（出力は含まず、各コードセルの前に日本語解説Markdownを挿入）。

| 枠 | コンペ | notebook | 原著者 | スコア |
|---|---|---|---|---|
| 固定 | Biohub - Cell Tracking During Development | [Kimi Notebook v17](https://www.kaggle.com/code/yunusgmsoy/kimi-notebook-v17) | YUNUS GÜMÜŞSOY | Public 0.918 / Best **0.923** |
| Playground | Playground S6E8 - Predicting Smartphone Addiction | [Predicting Smartphone Addict \| NN Residual Network](https://www.kaggle.com/code/anthonytherrien/predicting-smartphone-addict-nn-residual-network) | Anthony Therrien | **0.97123** |
| 実コンペ | RSNA Knee Abnormality Detection | [RSNA Knee CoAtNet Transformer Blend](https://www.kaggle.com/code/aldofhrzy/rsna-knee-coatnet-transformer-blend) | MUHAMMAD ALDO FAHREZY | **0.932** |

**本日の一言テーマ: 「スコアの出どころを、手法の質と切り離して読む」**
3本のうち2本（Playground・実コンペ）は、公開スコアのほぼ全てが**他人の提出物・他人の重み**に由来していました。パイプラインの作法としては学ぶ価値が大きい一方、数字をそのまま「この手法の実力」と読んではいけない、という日でした。

---

## 1. Biohub - Cell Tracking During Development / Kimi Notebook v17

- **原著者**: YUNUS GÜMÜŞSOY（[@yunusgmsoy](https://www.kaggle.com/yunusgmsoy)）
- **リンク**: <https://www.kaggle.com/code/yunusgmsoy/kimi-notebook-v17>
- **スコア**: Public 0.918 / Best 0.923（V7）・110 upvotes・Gold・実行56分（GPU T4 x2）
- **ローカル**: `biohub_kimi-notebook-v17.ipynb`

### 学べる主要テクニック

1. **変更点を1つに絞った実験設計**: 「fixed-90 dual-seed ベースライン（LB 0.913）に、モデルレベルの変更を1つだけ加える」と宣言し、その1つ（harmonic mutual-support association fusion）以外は一切触らない。差分が明確なので、スコアの動きを因果として読める。
2. **調和平均による双方向リンク融合**: 順方向（t→t+1）と逆方向（t+1→t）のリンク確率を調和平均で融合し、「片思いリンク」を落とす。調和平均は両方高いときだけ高くなる性質を持つ（F1スコアと同じ発想）。
3. **設定を環境変数に集約 + configuration guard**: 全ハイパーパラメータを `os.environ` に書き、直後のセルで期待値と一致するか検証して不一致なら即エラー。56分の推論を間違った設定で走らせる事故を最初の1秒で潰す。
4. **公式指標のローカル再実装（Cell 8-9）**: 学習データからホールドアウトを取り、主催者の metrics.md に沿った採点器を自前で実装。LB提出回数に縛られずに指標を直接最適化できる。しかもフラグ1つで無効化でき、本番の再現性を壊さない。
5. **PIPELINE MANIFEST（Cell 10）**: 「設定（要求）」ではなく「解決後の実際の状態」を1画面に印字する。`weights_found=False` なら dual-seed のつもりでも実体は single-seed。原著者はこの印字がなかったために過去に3件の静かなバグ（DeepCenterが一度もロードされない等）を見逃した、とコメントに明記している。

### 評価指標の要約

各時刻ごとに予測ノードと正解ノードを**物理距離（µm、ボクセル異方性 z=1.625 / y,x=0.40625 を換算）** でハンガリアン法により一対一マッチングし（半径7µm）、検出・親子リンク・分裂イベントを組み合わせて採点するトラッキング指標。分裂は主催者の exploit 対策パッチにより「単一連結成分 + 両娘の系譜カバー + アンカー + フォーク」を満たす必要があり、水増しが効かない。だから本notebookは検出閾値を 0.96875 と高く取り短トラックを捨てて精度を稼ぎ、ILP重みで「細胞が勝手に消える/湧く」ことを抑え、ギャップクローズで系譜の断絶を防ぐ、という**指標の構造に直接対応した設計**になっている。

### 改善点の考察

同コンペの上位公開notebookを見ると、0.96台（[biohub ct mix divaug](https://www.kaggle.com/code/pilkwang/biohub-ct-mix-divaug) 0.969、[Metric_hack_last_call_update](https://www.kaggle.com/code/boristown/dark-agi-biohub-cell-tracking-solution) 0.966 等）と 0.92 前後の「no-hack」系にはっきり二極化している。前者は指標の抜け穴を突く系統で、後者が実質的な手法競争のフロンティア。本notebookは後者の中では最上位（0.923）。

- **未採用の手法1: 学習側の多様性** — 本notebookは単一の学習済み重み（`unet_transformer/split_0`）が中心で、dual-seedも実体としては2つ目の重みが見つかっていない可能性がある（マニフェストが `weights_found` を印字しているのはそのため）。[Biohub Dual Seed Frame Retention Guard V1](https://www.kaggle.com/code/indarkarhana/biohub-dual-seed-frame-retention-guard-v1)（0.913）や [Biohub Top Notebook 0.913](https://www.kaggle.com/code/saitejabandaruin/biohub-top-notebook-0-913) が試みているように、**異なるシード/前処理で学習した検出器を実際に2本以上用意して融合する**余地がある。
- **未採用の手法2: TTAの拡張方向** — 現在は平面内の反転TTA（8視点相当）に留まる。z方向のフリップやスケールTTA、あるいは複数の検出閾値で推論して合議を取る「閾値アンサンブル」は未使用。
- **未採用の手法3: ILPの目的関数への指標の直接反映** — ILP重み（appearance / disappearance / division）は手でグリッド探索している。ホールドアウト採点器がすでにあるのだから、**ベイズ最適化（Optuna）でILP重みを自動探索**できるはず。原著者自身が「0.3/1.0/2.0/3.0 を試して全部0.915だった」と書いており、手探索の非効率が見える。
- **関連文献**: [Higher-Order Cell Tracking Transformer (HOCT)](https://arxiv.org/abs/2607.11754) は、リンク候補（エッジ）同士を3D幾何事前分布のもとで**互いにattentionさせる** edge-centric アーキテクチャで、深い事前学習画像エンコーダなしにSOTAを達成している。本notebookの「双方向融合」は2方向の合意を取る手作りのルールだが、HOCT の発想は**リンク候補集合を丸ごとモデルに考えさせる**方向であり、ILPの前段でより良い候補スコアを作れる可能性がある。また [CTransformer](https://arxiv.org/abs/2512.14472) は C. elegans の4D細胞形態アトラスを550細胞段階で80%精度で構築しており、胚発生という同種のタスクにおける最新の到達点として参考になる。
- **改善提案**:
  1. 別シード・別前処理で学習した検出器を実際に2本以上用意し、`weights_found=True` を満たしたうえで dual-seed 融合を機能させる（今は設定だけが dual-seed の可能性がある）。
  2. Cell 8-9 のホールドアウト採点器に Optuna を接続し、ILP重み・検出閾値・ギャップクローズ距離を同時に自動探索する（手探索の 0.3/1.0/2.0/3.0 のような粗いグリッドから脱する）。
  3. TTAをz方向フリップ・スケールに拡張し、さらに複数検出閾値の合議を取る。
  4. エッジスコアの段階で HOCT 型の edge-to-edge attention を導入し、ILPに渡す候補の質そのものを上げる。
  5. ホールドアウトは現在「種類ごとに2サンプル」と少ない。指標のばらつきが大きいため、サンプル数を増やすか、複数の分割で平均を取って**ローカル指標の信頼区間**を把握する（0.001の差を追う段階では必須）。

---

## 2. Playground S6E8 - Predicting Smartphone Addiction / NN Residual Network

- **原著者**: Anthony Therrien（[@anthonytherrien](https://www.kaggle.com/anthonytherrien)）
- **リンク**: <https://www.kaggle.com/code/anthonytherrien/predicting-smartphone-addict-nn-residual-network>
- **スコア**: Public 0.97123（本日時点の公開最高スコア帯）・36 upvotes
- **ローカル**: `playground_predicting-smartphone-addict-nn-residual-network.ipynb`（元は巨大な1セル。読みやすさのため論理単位で10セルに分割）

### 学べる主要テクニック

1. **リーク防止の徹底**: `train_test_split` を**先に**行い、`ColumnTransformer` を学習側のみに `fit`。検証・テストには `transform` だけ。コメントにも `# Split before fitting preprocessing to prevent leakage` と明記されている。
2. **out-of-fold スタッキング**: LightGBM を5-fold で回し、各foldの予測（OOF）を**1列の特徴量**としてNNの入力に追加。木モデルが得意な「軸に平行な閾値」をNNに渡し、両者の弱点を補う。検証・テストは5モデルの平均。
3. **残差MLP**: `features = features + residual` のスキップ接続で勾配消失を防ぎ、BatchNorm + Dropout で安定化。表形式データでは残差ブロック2つ程度の中規模が実用的。
4. **AUCで重みを選ぶ手動 early stopping**: 損失は `BCEWithLogitsLoss`（AUCは微分不可能なので直接最適化できない）だが、エポックごとに検証AUCを測り最良の重みを `copy.deepcopy` で保存・復元。`deepcopy` を忘れると重みが上書きされて静かにバグる。
5. **ブレンド時のid整合性チェック**: 行数だけでなく `np.array_equal(submission["id"], test_ids)` で**並び順まで**検証。ブレンド系notebookで最も多い事故を潰している。

### 評価指標の要約

2値分類の **ROC AUC**（「正例と負例をランダムに1つずつ選んだとき、正例に高いスコアを付ける確率」）。予測値の**順位のみ**を見るためキャリブレーション不要で、クラス不均衡でも「全部多数派」戦略が通用しない。だから最終ブレンドは確率の加重平均で済ませられる（単調変換なら順位は壊れない）。

### 改善点の考察

- **⚠️ 最大の論点**: 最終ブレンドの重みは `sub1=2.9, sub2=0.1, nn=1e-4`。**自分で学習したNNの寄与は事実上ゼロ**で、LB 0.97123 は外部提出 `sub1` のスコアとほぼ同一。タイトルは「NN Residual Network」だが、スコアを作っているのはNNではない。
- **他notebookとの比較**: 同コンペの上位公開notebookを確認したところ、[S6E8 Addiction LB 0.97113](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97113)（0.97123）は「モデル部分は非公開、提出CSVだけ共有」と明言しており、[\[S6E8\] TOP-1 PUBLIC 0.97099](https://www.kaggle.com/code/daniilkrasnovvv/s6e8-top-1-public-0-97099) は3本の公開提出を `rankdata` で順位平均するだけの4セルnotebookだった。**0.971台の公開notebookはほぼ全てが「公開提出の載せ替え・再ブレンド」で構成されている**というのが今日の実態。一方 [Why Every S6E8 Notebook Above 0.97110 Overfits](https://www.kaggle.com/code/szymonkapiski/why-every-s6e8-notebook-above-0-97110-overfits)（0.97115・8/20に扱った系統）は、この帯域がpublic LBへの過剰適合であると指摘している。残り8日、**Private LBでは0.971台が総崩れする可能性を織り込むべき**。
- **技術面で未採用の手法**: ①カテゴリ変数に one-hot ではなく **target encoding / embedding層** を使う（高カーディナリティで効く）、②単一のホールドアウト（10%）ではなく **k-fold でNN自体もOOFを作る**（現在NNの評価は1分割のみで、AUC推定のばらつきが大きい）、③LGBM以外（CatBoost・XGBoost）のOOFも特徴量に足す、④TabM / RealMLP のような近年の表形式向けNN。
- **関連文献**: [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/pdf/2410.24210) と [RealMLP](https://openreview.net/forum?id=fwajDrDy89) は、いずれも「MLPを丁寧に作り込めばGBDTと同等以上になる」ことを示した系統で、[TabArena](https://arxiv.org/pdf/2506.16791) のような近年のベンチマークでも TabM / RealMLP が GBDT と同等かそれ以上と報告されている。TabM は**1つのモデル内で安価に複数予測を出す（パラメータ効率的アンサンブル）**発想で、本notebookのような単一MLP + 外部ブレンドより筋が良い可能性がある。実際、同コンペには TabM / RealMLP 版のnotebook（各0.969前後）が公開されている。
- **改善提案**:
  1. まず**外部提出を一切混ぜない状態でのスコア**を測る。NN単体・LGBM単体・両者ブレンドの3点を出さないと、この設計の実力が分からない。
  2. NNの評価を単一ホールドアウトから 5-fold OOF に変え、AUCの標準偏差も併記する（0.001を争う帯域では1分割の推定は信頼できない）。
  3. カテゴリ変数を one-hot ではなく embedding 層に変え、`add_indicator` の欠損フラグと合わせて表現力を上げる。
  4. スタッキングの1段目を LGBM だけでなく CatBoost・XGBoost・RealMLP に広げ、多様性のあるOOF列を複数持たせる。
  5. Private LB を見据え、public LB 0.971台の載せ替えを追うのをやめ、**CVで選んだ自分の提出を1つは必ず選択枠に入れる**（8/20・8/23 の考察と同じ結論に到達している）。

---

## 3. RSNA Knee Abnormality Detection / RSNA Knee CoAtNet Transformer Blend

- **原著者**: MUHAMMAD ALDO FAHREZY（[@aldofhrzy](https://www.kaggle.com/aldofhrzy)）
- **リンク**: <https://www.kaggle.com/code/aldofhrzy/rsna-knee-coatnet-transformer-blend>
- **スコア**: Public **0.932**（本日時点で公開notebook中の最高）・実行2分18秒
- **ローカル**: `competition_rsna-knee-coatnet-transformer-blend.ipynb`

### 学べる主要テクニック

1. **rank blend（順位ブレンド）**: すべてのブレンドを確率ではなく `rank(method='average', pct=True)` による**パーセンタイル順位**で行う。評価指標がAUC（順位のみを見る）なので、キャリブレーションの違うモデル同士を公平に混ぜられる。確率のまま平均すると「0.99を連発する自信過剰なモデル」が実力以上に支配してしまう。
2. **所見ごとに違うブレンド設計**: E10段では10所見に `0.50×transformer + 0.50×RadImageNet` を適用しつつ、**ベーカー嚢腫と骨折だけは transformer をそのまま温存**（`_RAD_EXCLUDE`）。macro-AUC は所見ごとに1/12ずつの重みなので、平均で得しても希少所見で損をすると割に合わない、という指標構造を直接利用している。
3. **バックボーンの多様性**: DINOv2（自然画像・自己教師あり）、RadImageNet ResNet-50（医用画像で事前学習）、CoAtNet（畳み込み+attentionのハイブリッド）、独自transformer。誤りの相関が低いほど順位平均の効果が大きい。
4. **物理単位での前処理**: `CROP_MM = 130.0` とピクセルではなく**ミリメートルで切り出す**。DICOMの `PixelSpacing` を使って施設ごとの撮像条件差を吸収する、医用画像の基本作法。`SLICE_BAND` でスライス両端を捨てるのも同様。
5. **フォールバックと安全装置**: 「CoAtNet出力があればブレンド、なければ検証済みの既存提出をそのまま残す」という分岐、重みファイルの SHA-256 検証、`isfinite().all()` と値域チェック。9時間枠のCode Competitionで「全部落ちて0点」を防ぐ設計。
6. **名前空間の衝突回避**: 4つの独立パイプラインを1つのnotebookに同居させるため、`_rad_` / `_A5_` のような接頭辞で変数名を分離。Kaggleのnotebookは全セルが同じグローバル名前空間を共有するため、原始的だが確実な手段。

### 評価指標の要約

**12所見の macro-averaged ROC AUC**（`Final Score = (1/12)ΣAUC_i`）。所見ごとに独立してAUCを計算し単純平均するため、有病率の高い変形性関節症も稀な骨折も**同じ1/12の重み**を持つ。これは「稀だが見逃すと重大な所見を当てる能力」を評価に含めるための設計であり、同時に「所見ごとに最適なブレンド相手を選ぶ」という最適化の余地を生んでいる。順位しか見ない指標なので、rank blend が理論的に正当化される。

### 改善点の考察

- **⚠️ 最大の論点**: このnotebookは**自分では1つもモデルを学習していない**（実行時間2分18秒）。0.932 は「良いモデルを作った」結果ではなく「良い混ぜ方を見つけた」結果であり、**元の重みが公開され続ける限りにおいてのみ成立する**。
- **他notebookとの比較**: [Knee MRI: twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)（0.924）は**単一モデル・アンサンブルなし・TTAなし**でこのスコアを出しており、実力という意味ではこちらが上。[Bend the Knee to DINOv3 (ensembled)](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled)（0.922）や [RSNA Knee DINO-RadImageNet Rank Ensemble](https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble)（0.920・8/19に扱った）と比べると、本notebookはそれらの**上に乗っている**構造で、独自の学習成分がない。
- **未採用の手法**: ①**TTA が使われていない**（"no ensembling and no TTA" と明記された単体0.924モデルをそのまま使っている）。反転TTAだけでも上積みの余地がある。②**ラベル生成の改善**——構造化ラベルは58件のみで、残りはLLMがレポートを読んで作ったソフトラベル。複数LLMの合議や、58件の正解ラベルを使ったラベル品質の較正（キャリブレーション）は上位の分かれ目になりうる。③**Efficiency Track**——本コンペには実行時間を含む効率賞トラックがあり、2分18秒という本notebookの実行時間はむしろ効率賞に極めて有利。ただし重みの読み込み時間は含まれる点に注意。
- **関連文献**: 膝MRIのマルチラベル分類は [MRNet](https://stanfordmlgroup.github.io/projects/mrnet/)（Stanford, 1,370検査、スライスごとのCNN特徴を max pooling で集約）が原点で、近年は [KneeXNet](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full) のように**グラフ畳み込みでスライス間・断面間の空間依存性を明示的にモデル化**し、マルチスケール特徴融合と対照学習を組み合わせる方向が出ている。本notebookの「6スロット（断面×脂肪抑制）」は空間関係を手作業で整理したものだが、GCN で断面間の関係を学習させるのは自然な次の一手。また [SB-SSL](https://arxiv.org/pdf/2208.13923) は**スライス単位の自己教師あり事前学習**で少ラベル環境の膝MRI分類を改善しており、構造化ラベルが58件しかない本コンペの状況に直接刺さる。
- **改善提案**:
  1. まず**単体0.924モデルにTTA（左右反転・軽い回転）を足す**だけで上積みが出るか確認する。ブレンドを増やすより費用対効果が高い可能性がある。
  2. 58件の構造化ラベルを使って、LLM生成ソフトラベルの**信頼性を所見ごとに検証・較正**する。骨折のような希少所見でラベルノイズが大きいなら、macro-AUC への打撃は1/12では済まない（その列のAUCが大きく落ちる）。
  3. rank blend の重みを public LB ではなく**ホールドアウトのmacro-AUCで決める**。所見ごとにαを変える設計は自由度が高く、public LBに合わせるとそのまま過剰適合になる。
  4. 断面間・スライス間の関係を GCN やクロスアテンションで明示的に扱う（KneeXNet方向）。現在の6スロット構成は人手のヒューリスティック。
  5. 少ラベル対策として SB-SSL 的なスライス単位の自己教師あり事前学習を導入し、4,407件の画像を（ラベルなしでも）全部使う。
  6. Efficiency Track への提出も検討する。2分台の推論時間は効率スコアで大きな優位になる。

---

## まとめ

今日の3本は、偶然にも**「公開スコアの読み方」**という一つのテーマで揃った。

- **Biohub（0.923）** は、単一の変更点・自前の公式指標再実装・PIPELINE MANIFEST と、**何が効いたのかを自分で検証できる作りになっている数少ない例**だった。「設定を印字するだけでは足りない、解決後の実際の状態を印字せよ」という教訓は、どんなパイプラインにも持ち帰れる。
- **Playground（0.97123）** は、丁寧なリーク防止とOOFスタッキングを実装していながら、最終的に自分のNNに重み `1e-4` しか与えていなかった。**タイトル・スコア・実際の寄与が食い違っている**典型例。
- **RSNA（0.932）** は、rank blend と所見ごとのα設計という**指標構造を正しく利用した優れた工夫**を持つ一方、自分では1つもモデルを学習しておらず、公開重みの上に乗っている。

いずれも学ぶ価値は確かにあるが、**「学ぶ対象」と「信じてよいスコア」は分けて読む**必要がある。Playground S6E8 は残り8日、Private LB でこの帯域がどう崩れるかは、この教訓の実地検証になる。
