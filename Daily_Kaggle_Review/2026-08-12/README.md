# 2026-08-12 Kaggle日次レビュー

固定コンペ「Biohub - Cell Tracking」1本 + Playground Series 1本 + 開催中の実コンペ1本、計3本を解説付きipynb化しました。

## 1. Biohub - Cell Tracking During Development

- コンペ: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- notebook: [Biohub Solution](https://www.kaggle.com/code/kaiwalyaatulraut/biohub-solution) by Kaiwalya Raut
- スコア: コード一覧のBest Score表示で0.966（notebook詳細ページ自体のPublic Score表示は0.885 V1）

**主要テクニック**
- ガウシアン平滑化＋局所ピーク検出（`peak_local_max`）による古典的な3D細胞検出
- ハンガリアン法（`linear_sum_assignment`）による物理距離ベースのフレーム間対応付け
- 距離しきい値ベースの明示的な細胞分裂検出ルール
- 深層学習（UNet+Transformer）による事前学習済みモデルも併用し、ILP（整数計画法）ソルバーでグラフ全体を最適化

**評価指標の要約**: `score = adjusted_edge_jaccard + 0.1 × division_jaccard`。フレーム間リンク（edge）の正しさと細胞分裂検出の正しさを合成した指標で、ノード数の過剰予測にはペナルティが課される。

**改善点の考察**
- 同コンペのコード一覧を見ると、上位陣（`biohub-v6-ultra-best`, `Biohub Harmonic Bidirectional Association V1`など）は深層学習ベースのUNet3D+Transformerパイプラインを主軸にしており、今回選んだnotebookのような古典的画像処理（ガウシアン平滑化+ピーク検出）だけでは検出漏れ・過検出のバランスで劣る可能性がある。一方、今回のnotebookはILPソルバーやTTA関連のコードも同梱しており、実質的にハイブリッド構成になっている。
- コード一覧には`Metric_hack_last_call`のような「メトリックの穴を突く」タイプの高スコアnotebookも複数存在した。これらは意図的に除外し、手法として理解しやすいものを選定している。
- 最新研究では、Highigher-Order Cell Tracking Transformer（HOCT）のように、候補となる細胞リンク同士がAttentionで直接やり取りする「edge-centric」なアーキテクチャや、Cell-TRACTR・Trackastraのようなend-to-endのTransformerベース追跡モデルが提案されている。これらは事前学習済み画像エンコーダなしでも高い精度を達成しており、今回のnotebookのような「検出→ハンガリアン法でリンク」という2段階パイプラインの限界（検出誤りが下流のリンク推定に伝播する）を克服する方向性として参考になる。
- 改善提案:
  1. 分裂検出のしきい値（`DIV_PARENT_DIST`, `DIV_SISTER_DIST`）を固定値ではなく、局所的な細胞密度に応じて動的に調整する。
  2. ギャップクロージング（1フレーム欠損の補完）を2フレーム以上の欠損にも拡張する。
  3. HOCTのようなedge-centric Transformerによる検出・リンクの同時最適化を試す。
  4. TTA（Test-Time Augmentation）のバリエーションをアンサンブルし、検出の頑健性を上げる。
  5. Division Jaccardの重み(0.1)が小さいため、分裂検出の改善が全体スコアに与える影響を定量的に事前評価してから工数を配分する。

## 2. Playground Series S6E8: Predicting Smartphone Addiction

- コンペ: [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)
- notebook: [S6E8: mix the meta-models, then fix the weak bands](https://www.kaggle.com/code/raykkretzschmar/s6e8-mix-the-meta-models-then-fix-the-weak-bands) by Rayk Kretzschmar
- スコア: Public Score 0.97095（Best Score、Version 17）

**主要テクニック**
- 74モデルの公開OOFライブラリの上に、因子分解機（Factorization Machine）を3種類追加してアンサンブルの多様性を向上
- 2種類のメタモデル設計（グローバル・ロジスティック回帰 vs. 欠損パターン考慮のregime設計）を「選ぶ」のではなく「パーセンタイルランクで混ぜる」
- 精度が弱い特定の値域（バンド）だけに特化したFMによる局所補正（グローバルな順位を壊さない設計）
- 誠実なOut-of-Fold評価と、リーダーボードのノイズ（ブートストラップ標準偏差）を踏まえた統計的に慎重な意思決定

**評価指標の要約**: ROC-AUC。予測確率のランキング（順位）の質のみに依存する指標で、モデルの出力スケールの違いに影響されない。

**改善点の考察**
- 同コンペのコード一覧では、47〜94モデルのスタッキング（`Stacking 47 Models`, `S6E8 Ultimate Ensemble Stacking 50 Models`など）が多数見られ、今回選んだnotebookも最終的に94メンバーへ拡張している。一方、`Everything above 0.970 is inside the noise floor`という別notebookのタイトルが示す通り、コミュニティ全体でスコアがノイズフロアに漸近している状況が伺える。
- 今回選んだnotebookが採用していない手法として、コード一覧上位には「exact-value/ORIGデータセットのXGBoost」（Kodai Fukuda作、このnotebook内でも言及・検証されているがOOF改善もLB改善は伴わず不採用）や、LightGBMをメタモデルに使う手法（このnotebook内の「Negative results」セクションで劣ることが実証済み）がある。
- 因子分解機（FM）やDeepFM系のモデルはKaggleのタブularコンペで定番になりつつあり、最近の研究ではTabM（パラメータ効率的なアンサンブル）やTabPFN-2.5のようなタブラー基盤モデル（事前学習済みモデルを使い勾配ブースティングに匹敵する性能を出す）が登場している。今回のnotebookのアンサンブルにこうした基盤モデルを新メンバーとして追加することで、既存メンバーとの相関がさらに低い予測を得られる可能性がある。
- 改善提案:
  1. TabPFNやTabM等の新しいタブラー基盤モデルを、既存パックとの順位相関を測定した上でライブラリに追加する。
  2. バンド局所補正を適用するバンドの選定基準（gap<0.015で採用）を、より細かい値域分割で再検証する。
  3. 74/94メンバーのうち相関が極めて高い（>0.998）メンバーを整理し、計算コストを削減しつつ性能を維持できるか検証する。
  4. メタモデルの2つの設計（global/regime）以外に、木ベースモデルによる非線形メタモデルを、正則化を強めに設定した上で再検証する（このnotebook内ではLightGBMメタは劣る結果だったが、別のアーキテクチャの余地はある）。
  5. 一部の負の結果（DCN-styleクロスネットワーク、isotonic較正など）が「なぜ効かなかったか」の理論的な理解を深め、次にどのアプローチなら効きそうかの仮説を立てる。

## 3. RSNA Knee Abnormality Detection

- コンペ: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- notebook: [RSNA Knee Solution](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-solution) by Kaiwalya Raut
- スコア: Public Score 0.899（Best Score、Version 3）

**主要テクニック**
- 事前学習済みDINOv2（Vision Transformer）バックボーン＋撮像スロット単位の部分的ファインチューニング（最終層のみ開放）
- DICOMメタデータの物理座標（`ImagePositionPatient`/`ImageOrientationPatient`）からのスライス順序・左右（laterality）の頑健な復元
- 診断項目ごとに異なる撮像面への注意重み（`SlotHead`のAttention＋事前知識バイアス）
- 推論時のオーバーラップTTA窓と、診断項目別プーリング戦略（max/top2/mean）
- 監査ゲート付きの複数モデルファミリー（DINOv2＋EfficientNet-B3）アンサンブル
- フェイルセーフな提出（常に0.5埋めのベンチマーク提出を先に書き出す）

**評価指標の要約**: 12種類の膝異常所見それぞれのROC-AUCのマクロ平均。稀な所見（骨折など）も多い所見と同じ重みで評価される。

**改善点の考察**
- 同コンペのコード一覧には、放射線科レポート（テキスト）を使う`RSNA Knee +90% reports LLM 30 epochs`という手法が存在するが、今回選んだnotebookは画像のみを使っており、コンペが提供する「画像＋レポートのペア」というマルチモーダルな特性を活かし切れていない。
- `RSNA Knee | Data structure, EDA, baseline`のような、より丁寧なEDAに特化したnotebookもあり、今回選んだnotebookは推論本体に重点を置いているぶん、データ分布の可視化（EDA）は簡潔。
- 最新研究では、MM-DINOv2のようにDINOv2を元々マルチモーダル医用画像解析に適応させる手法や、Medical Slice Transformer（MST）のように2D DINOv2特徴を3Dボリュームに拡張しつつ説明可能性（サリエンシーマップ）を高める手法が提案されている。今回のnotebookが用いる「複数スライスをスロットとして扱いAttentionで統合する」設計は、こうした最新のMST的アプローチと方向性が近く、放射線科レポートのテキスト情報を統合すればさらなる改善余地がある。
- 改善提案:
  1. 放射線科レポートのテキストをテキストエンコーダ（例: 医療特化のBERT系モデル）で埋め込み、画像特徴と結合するマルチモーダル化。
  2. `SLOT_PRIOR_TABLE`（診断項目×スロットの事前知識）を、実際のOOF評価に基づいて再検証・微調整する。
  3. EfficientNet-B3ブレンドの監査条件（`ALLOW_UNAUDITED_B3`）をより多くのデータで検証し、ブレンド比率の最適化を行う。
  4. MM-DINOv2やMedical Slice Transformerのような、より医用画像に特化したバックボーンとの比較実験。
  5. 稀な所見（骨折・Baker's cystなど）に対するクラス不均衡対策（オーバーサンプリングや損失関数の重み付け）を追加し、マクロ平均AUCで不利になりやすい少数派ラベルの精度を底上げする。

## まとめ

今回は「古典的画像処理×ILP」「アンサンブル理論の丁寧な検証」「医療画像の前処理の作法」という3つの異なる側面を学べる構成になった。特にS6E8のnotebookは、Kaggleのアンサンブル手法として「リーダーボードのノイズを正しく理解した上で意思決定する」という統計的な誠実さの好例として、他の2本にも応用できる姿勢だと感じた。

## NotebookLM音声・動画解説

本日分のREADMEをソースに、NotebookLM（Gemini Notebook）で音声解説・動画解説を作成し、`notebooklm/`フォルダに保存しました（GitHubにはアップロードしていません）。
