# 2026-08-14 Kaggle日次レビュー

今日は固定枠のBiohub 1本、Playground Series S6E8から1本、実コンペ（RSNA Knee Abnormality Detection）から1本、計3本の高スコアnotebookを解説付きでipynb化した。

## 1. Biohub - Cell Tracking During Development

- **コンペ**: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- - **notebook**: [Biohub 11](https://www.kaggle.com/code/nikitagajbhiye30/biohub-11) by Nikita（nikitagajbhiye30）、Public/Best Score 0.915（Bronze、56 upvotes）
  - - **ファイル**: `biohub_biohub-11.ipynb`
   
    - **学べる主要テクニック**
    - - 学習済みUNet3Dによる3D+timeの細胞中心検出と、Transformer的特徴量を使うnode/edgeスコアリング（node-transformer）の組み合わせ
      - - 整数計画法（ILP）によるフレーム間の1対多対応（細胞分裂）を含む対応付け最適化
        - - ギャップクロージング（検出漏れによるトラック断裂の復元）と、分裂の幾何学的安全制約（親子・姉妹細胞間の距離上限）による後処理
          - - 環境変数ベースの設定管理により、実行のたびに「何を変えたら何が起きたか」を追跡できるようにする実務的な書き方
           
            - **評価指標の要約**: score = adjusted_edge_jaccard + 0.1 × division_jaccard。フレーム間リンクの正しさ（edge Jaccard）と、稀少だが重要な細胞分裂イベントの正しさ（division Jaccard、0.1倍のボーナス重み）を別々に評価する合成指標。本notebookはILPの分裂用エッジ重みと`SAFE_DIV_*`系パラメータで division_jaccard の適合率を、ギャップクロージング/モーション再リンクで edge_jaccard の再現率を、それぞれ個別に狙う設計になっている。
           
            - **改善点の考察**
            - 1. **他notebookとの比較**: 同コンペのBest Scoreソートを見ると、上位の多くが「Biohub 159B」という共通の固定ベースライン設定から派生し、GPTを使った出力監査（既知のベースライン出力とのノード数・エッジ数・分裂数の差分チェック）を組み込んでいる点が特徴的だった。本notebookも同様の監査ステップを持つが、他の上位notebook（例: `biohub-v6-ultra-best`、過去日に扱用済み）と比べると、今回選んだものはDeepCenterの信頼度veto（`DEEPCENTER_*`系）を無効化した設定になっており、その分adaptive short-track rescueなど他の安全弁に依存する設計になっていた。
              2. 2. **関連文献の確認**: 過去のPaper Digestで扱ったCell-TRACTR論文（DETRベースのend-to-end検出+追跡）やCell-HOTA指標との関連で、現在の細胞追跡研究では「検出→ILP対応付け→後処理ヒューリスティック」という多段パイプラインから、track queryによるend-to-end学習への移行が試みられている。本notebookのような多数のヒューリスティックパラメータへの依存は、解釈性・調整の柔軟性という利点がある一方、Cell-TRACTR系のアプローチが目指す「後処理レス」の方向性とはトレードオフの関係にある。
                 3. 3. **改善提案**:
                    4.    - ILPのエッジ重み（appearance/disappearance/division）をハードコードするのではなく、Optunaなどでnested CV相当のオフライン評価に対してチューニングする余地がある。
                          -    - GPT監査ステップが比較しているベースライン（Biohub 159B）自体の妥当性を、複数の独立したベースラインに対してクロスチェックすると監査の信頼性が上がる。
                               -    - `SAFE_DIV_*`のような固定しきい値パラメータを、フレームごとの細胞密度に応じて動的に調整すれば、密集領域での偽陽性分裂を減らせる可能性がある。
                                    -    - HOTA/Cell-HOTA的な指標でDetA/AssA/DivAを個別に計測し、どのサブスコアがボトルネックかを可視化すると、次にどのパラメータ群を優先的にチューニングすべきかが明確になる。
                                     
                                         - ## 2. Predicting Smartphone Addiction (Playground Series S6E8)
                                     
                                         - - **コンペ**: [Predicting Smartphone Addiction (Playground Series S6E8)](https://www.kaggle.com/competitions/playground-series-s6e8)
                                           - - **notebook**: [The strongest fully-reproducible stack - LB 0.9708](https://www.kaggle.com/code/dariushafshar/the-strongest-fully-reproducible-stack-lb-0-9708) by Dariush Afshar（dariushafshar）、Best Score 0.97083
                                             - - **ファイル**: `playground_the-strongest-fully-reproducible-stack-lb-0-9708.ipynb`
                                              
                                               - **学べる主要テクニック**
                                               - - 81個の公開OOFモデルをプールし、ライセンス不明なメンバーを機械的に除外する「ライセンスゲート」
                                                 - - 各メンバーをランクとロジットの2表現でメタモデルに渡す dual representation
                                                   - - 欠測パターン・メンバー間不一致度を特徴量にした regime-interaction スタックとの固定比率(1/3)ランク混合
                                                     - - nested honest CV（学習に使っていないfoldだけでスコアする）による、効いた工夫・効かなかった工夫の統計的な検証（t値・全fold正か）
                                                      
                                                       - **評価指標の要約**: `addicted_label`のROC-AUC。ランキングの正しさだけを評価する指標であるため、本notebookは予測値そのものではなく正規化ランクを軸にメンバーを混合し、AUC最適化と自然に噛み合う設計にしている。
                                                      
                                                       - **改善点の考察**
                                                       - 1. **他notebookとの比較**: 同コンペのBest Scoreソート上位（`[S6E8] TOP-1 PUBLIC 0.97099`など）は、既存の複数notebookの`submission.csv`をそのままランク平均する「アンサンブルのアンサンブル」が多く、スコアはわずかに高いものの手法としての説明可能性は低い。本notebookはOOFレベルから honest CV で検証しており、スコアはわずかに劣るが再現性・信頼性の面で優れている。
                                                         2. 2. **関連文献の確認**: スタッキングにおける「メンバーの表現方法（ランク/ロジット/確率）の選び方」は、古典的にはVan der Laanらのsuper learner文献などで議論されてきたテーマで、近年はKaggle notebookで実証的に「複数表現を並列に与えてメタモデルに選ばせる」という素朴だが効果的なアプローチがよく使われる。
                                                            3. 3. **改善提案**:
                                                               4.    - 混合比率1/3は他プールからの継承値であり、in-fold最適化しつつ過学習コストを測定する「プラシーボ対照付きのfold内導出」を試す価値がある。
                                                                     -    - regime特徴量は欠測フラグ＋不一致度のみ。特徴量ごとの欠測パターン（MCAR/MARの違い）を明示的に扱うモデルとの組み合わせが未検証。
                                                                          -    - プールメンバー間の相関が0.99と非常に高く飽和状態にあるため、新規メンバー追加よりも「既存メンバーをどう読むか」の工夫（例えば非線形なメタモデルだが正則化を強くしたもの）に投資する方が有望というのが著者自身の結論であり、追試の価値がある。
                                                                               -    - プライベートリーダーボードでこの1/3混合の利得が本当に残るかどうかは、著者自身が「限界」として明言しており、要フォローアップ。
                                                                                
                                                                                    - ## 3. RSNA Knee Abnormality Detection
                                                                                
                                                                                    - - **コンペ**: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
                                                                                      - - **notebook**: [Bend the Knee to DinoV3 (ensembled)](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled) by Mattia Angeli（mattiaangeli）、Public/Best Score 0.910（Bronze、36 upvotes、Version 25/29）
                                                                                        - - **ファイル**: `competition_bend-the-knee-to-dinov3-ensembled.ipynb`
                                                                                         
                                                                                          - **学べる主要テクニック**
                                                                                          - - 放射線科レポート文章を多言語対応（英・独・仏・西・希・露・トルコ語など）で解析するルールベース臨床NLP
                                                                                            - - 前置/後置の否定表現検出（NegEx的手法）と、否定語の作用範囲（文字数ウィンドウ）・転換接続詞による打ち消しの扱い
                                                                                              - - Grade表記（ローマ数字含む）・重症度語彙からの連続値スコア化（0/1ではなく確信度付きスコア）
                                                                                                - - DINOv2・RadImageNet事前学習ResNet-50による画像予測とのマルチモーダル（テキスト＋画像）アンサンブル
                                                                                                 
                                                                                                  - **評価指標の要約**: 12ターゲットのROC-AUCのマクロ平均。稀な所見（骨折など）も頻度の高い所見（関節液貯留など）と対等に評価されるため、本notebookは全12所見に同じ精度のNLPロジックを適用し、かつテキストが曖昧な場合は確信度を下げて画像モデル側に判断を委ねる設計で、稀な所見の見逃しを防いでいる。
                                                                                                 
                                                                                                  - **改善点の考察**
                                                                                                  - 1. **他notebookとの比較**: 同コンペのBest Scoreソート上位には`RSNA Knee +90% reports LLM 30 epochs`のようにLLM（大規模言語モデル）でレポートを直接分類する手法も見られた。本notebookのルールベースNLPは解釈性・再現性・実行コストの面で優れるが、言い回しの多様性への頑健さではLLMベースの手法に劣る可能性がある。
                                                                                                    2. 2. **関連文献の確認**: 臨床テキストの否定検出は古典的にはNegEx（Chapman et al.）が有名で、近年はBERT系の文脈埋め込みを使った否定・不確実性検出（ConText等）に発展している。本notebookの正規表現ベースの手法は、BERT系より軽量・高速だが、複雑な入れ子構造の否定（"no evidence of a tear, although mild edema is present"のような部分否定）への対応は限定的とみられる。
                                                                                                       3. 3. **改善提案**:
                                                                                                          4.    - 否定検出の精度を、実際のアノテーション付きサブセットで定量評価（precision/recallを個別に測定）し、言語ごとの精度差を把握する。
                                                                                                                -    - `NEG_WINDOW=90`のような固定の文字数ウィンドウを、文の構造（句読点位置）に基づく動的なウィンドウに置き換えると誤判定を減らせる可能性がある。
                                                                                                                     -    - テキスト予測の確信度（`conf`）を画像モデルとのブレンド重みに明示的に使っているかを検証し、テキスト・画像それぞれの寄与度をアブレーションで測定する。
                                                                                                                          -    - マクロ平均AUCの特性上、稀な所見（骨折・Baker嚢胞など）でのエラー分析を優先的に行うと、全体スコアの伸びしろが大きい可能性がある。
                                                                                                                           
                                                                                                                               - ## まとめ
                                                                                                                           
                                                                                                                               - 3本を通じて共通するテーマは「複合的な評価指標（Biohubのedge+division、RSNAの12ターゲットマクロ平均）に対して、パイプラインの各構成要素をどの部分指標に効かせるかを意識して設計する」という点だった。特にPlaygroundのnotebookが実践していた「nested honest CVで効いた工夫と効かなかった工夫を統計的に区別する」姿勢は、他の2本（パラメータの効果を体系的に検証しているわけではない）にも応用できる、汎用性の高い実務スキルだと感じた。
                                                                                                                               - 
