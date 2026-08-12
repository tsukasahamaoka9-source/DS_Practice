# Daily Kaggle Review — 2026-08-13

今日は固定コンペ1本・Playground Series 1本・実コンペ1本の計3notebookを解説付きで写経しました。

## 1. Biohub - Cell Tracking During Development

- **コンペ**: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- **notebook**: [Biohub Competition Solution](https://www.kaggle.com/code/biohub-competition-solution)
- **ファイル**: `biohub_biohub-competition-solution.ipynb`

**技術ポイント**
- UNet3D + Transformerによる細胞検出・追跡モデルと、`tracksdata.solvers.ILPSolver`を使ったグラフ最適化（整数線形計画法でフレーム間の対応関係を大域最適化）
- Test-Time Augmentation（4/8/9通りのフリップ・回転で推論しアンサンブル）による予測の安定化
- サイン・コサインを使った位置エンコーディングで細胞の空間座標をモデルに与える設計
- ギャップクロージング（一時的に検出が途切れた追跡を後処理でつなぐ）
- 提出前の合成フォレスト拡張（`augment_dataset`によるボーダーラインな"metric hack"寄りの後処理、notebook側もこれを明記）

**評価指標**: `score = adjusted_edge_jaccard + 0.1 × division_jaccard`。edge Jaccardは`TP / (TP + FP + FN)`をベースに予測総ノード数と正解総ノード数の乖離でペナルティを与える調整版、division Jaccardは細胞分裂イベントの検出精度を局所ウィンドウで評価する。

**改善分析**: 同コンペの他上位notebookと比較すると、本notebookはILPによる大域最適化と3D UNet検出の組み合わせが主軸で、これは2025年発表の`Ultrack`（Nature Methods, スケール横断的な細胞追跡フレームワーク）とも設計思想が近い。一方で、Higher-Order Cell Tracking Transformer（2026）のような、ILPに頼らずTransformerで直接高次の追跡関係を学習する手法も登場しており、ILPの計算コストがボトルネックになる密な細胞集団では有望な代替になりうる。改善案としては、(1) ILPソルバーの目的関数に division_jaccard の重み(0.1)を直接反映させて評価指標と最適化目標を一致させる、(2) TTAの回転角度をランダム化してモデルの向き不変性の弱点を洗い出す、(3) ギャップクロージングの許容フレーム数をvalidation setでグリッドサーチする、(4) 合成フォレスト拡張がスコアに与える寄与を消してtrue skillとの差分を確認する、の4点が有効と考える。

## 2. Playground Series S6E8: Predicting Smartphone Addiction

- **コンペ**: [Predicting Smartphone Addiction (Playground Series S6E8)](https://www.kaggle.com/competitions/playground-series-s6e8)
- **notebook**: [S6E8 | U Smart Phone Addict?](https://www.kaggle.com/code/anhadmahajan06/s6e8-u-smart-phone-addict) by Anhad Mahajan
- **ファイル**: `playground_s6e8-u-smart-phone-addict.ipynb`

**技術ポイント**
- ファイル名のスコア文字列を正規表現で自動検出し、新しい提出ファイルを追加するだけでコード変更なしにアンサンブルへ組み込める設計（"Future-Proof"）
- `scipy.stats.rankdata`で予測値を順位に変換してからブレンドする、ROC-AUCと相性の良いランクベース手法
- Linear Anchor・Power Rank Decay・Top3平均・幾何平均（`gmean`）・Sharp Power Blendの5種類の異なるブレンド戦略を並行生成

**評価指標**: ROC-AUC（順位の質のみに依存、絶対確率値には依存しない）。ランク変換してからブレンドする設計はこの指標の性質に直接対応している。

**改善分析**: MLWaveのKaggle Ensembling Guideや2025年のNVIDIA Kaggle Grandmasters Playbookでも、AUCのような順位ベース指標には単純平均よりランク平均・幾何平均が有利であることが繰り返し指摘されており、本notebookの設計はこのセオリーに沿っている。改善案としては、(1) 5種のブレンドを検証セットのAUCで自動選択・重み最適化する仕組みに発展させる、(2) 幾何平均を使うTop-5選択を固定値でなくvalidationベースで動的に決める、(3) GPUベクトル化で数千通りの重み組み合わせを探索する2025年の実践例のように、ブレンド重みの網羅的な探索を導入する、(4) モデル間の予測相関を見て「多様性の低い組み合わせ」を除外するロジックを足す、の4点が有効。

## 3. RSNA Knee Abnormality Detection（実コンペ）

- **コンペ**: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- **notebook**: [RSNA Knee — V2 Fold-Balanced Robust Ensemble](https://www.kaggle.com/code/sakhawathossen/rsna-knee-enhanced-ensemble) by Sakhawat Hossen（Tony Liのnotebookからのフォーク）, Public Score 0.899
- **ファイル**: `competition_rsna-knee-enhanced-ensemble.ipynb`

**技術ポイント**
- DICOMヘッダの並列読取と、`ImagePositionPatient`/`ImageOrientationPatient`から患者座標系の左右（laterality）を幾何学的に復元するロジック（タグ欠損時のフォールバック）
- 公式メタデータとの答え合わせによる、自前のT1/T2/PD・脂肪抑制判定ロジックの品質監査
- 医学的知見（`SLOT_PRIOR_TABLE`）をアテンションの初期バイアスに注入した`SlotHead`による、スロット単位のアテンション・プーリング
- DINOv2バックボーンの最終Nブロックのみを解凍する部分的ファインチューニング
- fold-balancedなランクベース最終アンサンブル（`FINAL_MODE`/`B3_TARGET_ALPHAS`による部位別ブレンド制御）

**評価指標**: 12種類の所見それぞれについてROC-AUCを計算し、その単純平均（macro-average）を取る。ターゲットごとに有病率が大きく異なるため、`POOL_PARTS`による部位別の特徴集約設計がターゲットごとの学習しやすさのばらつきに対応している。

**⚠️ 抽出上の注記**: 元notebookはページのテキスト抽出時にコードのインデント（字下げ）情報が失われるため、著者の意図を読み解いて手動で再構築しています。また元ページ自体が「78,289文字中50,000文字」で切り詰められていたため、最終セル（重みファイルの整合性検証ロジック）は未完成のまま原文が途切れており、推測で埋めることはせず本notebook内でその旨を明記した上で省略しました。

**改善分析**: 2025年のDINOv2医療画像応用（MM-DINOv2、DinoAtten3Dなど）と比較すると、本notebookの「最終Nブロックのみ解凍」というアプローチは、フルファインチューニングとLoRAのような軽量アダプタ手法の中間的な選択で、データ量が限られる医療画像コンペでは妥当なバランス。ただしLoRAは計算・メモリを35%程度削減しつつ精度低下も小さいとする報告もあり、対象を広げる余地がある。改善案としては、(1) 最終ブロック解凍の代わりにLoRAを試して学習効率と精度のトレードオフを比較する、(2) DinoAtten3Dのようなソフトアテンションによる3D全体プーリングと、本notebookのスロット単位プーリングを比較検証する、(3) `SLOT_PRIOR_TABLE`のバイアス値をアブレーション（除去して学習）し、事前知識注入の実際の寄与を定量化する、(4) fold-balancedアンサンブルの重みをvalidation AUCで自動最適化する、(5) 公式メタデータ突合の一致率が低いスロットに絞って正規表現ルールを再チューニングする、の5点が有効。

## まとめ

3本を通じて「ドメイン知識をモデル設計に事前注入する」（Biohubのsin/cos位置エンコーディング、RSNAのSLOT_PRIOR_TABLE）、「評価指標の性質に最適化手法を合わせる」（RSNA/PlaygroundのランクベースアンサンブルとROC-AUCの相性）という2つの共通パターンが見られた。特にPlaygroundとRSNAはどちらもROC-AUC系の指標であり、順位ベースのブレンドという同じ発想が異なるスケールのタスク（提出ファイル単位 vs マルチターゲット予測単位）で再利用されている点が興味深い。

本日の音声解説・動画解説はNotebookLMで生成し、`Daily_Kaggle_Review/2026-08-13/notebooklm/`にローカル保存します（GitHubへはアップロードしません）。
