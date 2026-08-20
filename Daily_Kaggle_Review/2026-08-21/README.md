# Kaggle日次レビュー 2026-08-21

今日の3本を貫くテーマは **「モデルの外側で勝負が決まる」** でした。
Biohubは**復元する仕組みと却下する仕組みを別モデルで持つ**、
RSNAは**正規表現による多言語レポート解析とDICOM幾何の復元**、
そしてPlaygroundは**合成データの格子構造をルックアップ表に変換する**。
3本とも、勾配降下法が触れない場所——前処理・後処理・集約の設計——に一番の工夫がありました。

Playground枠については、**あえて最高スコアのnotebookを選びませんでした**。
このコンペは終盤に入り、公開LB上位帯（0.9709〜0.9712）はほぼ全部が
**他人の`submission.csv`をランク平均するだけの5セルnotebook**になっています。
実際に候補だったPublic 0.97099の「TOP-1」notebookを開いて確認しましたが、実質4行でした。
代わりに選んだ0.96464のnotebookには、**再現可能なバグが1つ埋まっていて**、
それを追うほうが遥かに学びが多い状態です（後述）。

| 枠 | コンペ | notebook | スコア |
|---|---|---|---|
| 固定 | Biohub - Cell Tracking | [Biohub - Track Your Cells Development](https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells-development) | Public 0.916 |
| Playground | Predicting Smartphone Addiction (S6E8) | [📱 Robust Ensembling & Target Encoding Pipeline](https://www.kaggle.com/code/koushikkumardinda/robust-ensembling-target-encoding-pipeline) | Public 0.96464 |
| 実コンペ | RSNA Knee Abnormality Detection | [RSNA Knee: Take Care Of Your Knee](https://www.kaggle.com/code/anhadmahajan06/rsna-knee-take-care-of-your-knee) | Public 0.914 |

---

## 1. Biohub - Cell Tracking：復元する仕組みと、却下する仕組みを分ける

**原著者**: ANHAD MAHAJAN ([@anhadmahajan06](https://www.kaggle.com/anhadmahajan06))・76 votes（Silver）・Apache 2.0
**リンク**: https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells-development
**スコア**: Public 0.916 / Best 0.916（V12）・ランタイム 23分5秒（T4 x2）

### 学べる主要テクニック

- **DeepCenter 拒否権（veto）**。後処理で細胞を「復元」するたびに、
  **独立に学習した中心検出U-Netのヒートマップに問い合わせ**、スコアが低ければ復元を却下する。
  増やす処理（分裂修復・ギャップ埋め）と、それを抑えるブレーキを**別のモデルで持つ**設計。
  後処理でFNを減らすとFPが増えるのは避けられないので、**増やす側だけ強化しても指標は伸びない**。
- **トポロジー不変のラインフィット平滑化**。3フレーム窓で直線を当て、重み0.85で座標を引き寄せる。
  誰と誰がつながっているか（グラフの形）は**一切変えない**ので、
  エッジのスコアを壊さずにノードのマッチング（7.0µm判定）だけを改善できる。
  ここで `len(prev_ids) != 1` で打ち切り、**分裂点をまたいで平滑化しない**のが要点。
  分裂前後は座標が本来急変する場所なので、均すと逆効果になる。
- **合成中間ノードの輝度重心リファイン**。ギャップを埋めるために作ったノードは最初ただの中点。
  実画像の小窓を切り出し、20パーセンタイルを背景として引いてから
  **輝度重み付き重心**へ寄せる。ただし `GAP_REFINE_MAX_SHIFT_UM` を超えて動いたら**補正自体を却下**。
- **速度事前分布つき2段階モーション再リンク**。直前フレームの速度から次の位置を予測し、
  厳しいゲート(6.5µm)で確実なペアを先に確定 → 余りだけ緩い条件で救済。
  幾何（距離）と学習（エッジ確率）の**両方の証拠**でリンクを決める。
- **分裂修復の総量規制**。`SAFE_DIV_GLOBAL_FRAC_CAP` で追加できる分裂を全辺数の一定割合に制限。
  後処理の暴走に**上限で蓋をする**。

### 評価指標の要約

CTC系の複合スコア。**ノード検出精度＋エッジのJaccard＋分裂イベントの正誤**の合成で、
ノードのマッチングは **7.0µm のユークリッド距離**（voxel: z=1.625, y=x=0.40625µm）で判定される。
つなぎ忘れ(FN)も余計なリンク(FP)も両方減点されるため、
「沢山つなぐ」戦略と「自信のあるものだけつなぐ」戦略のどちらも、一方に偏ると損をする。
フレーム単位の検出精度（mAP等）では**軌跡の一貫性と分裂という構造変化**を測れないので代替にならない。

### 改善点の考察

**他notebookとの比較**（Codeタブを`?searchQuery=biohub`で一覧確認）

| notebook | スコア | 本notebookが採用していない要素 |
|---|---|---|
| biohub ct mix divaug | 0.969 | メトリックハック（座標範囲・時刻の検証漏れを突く） |
| Biohub Cell Tracking Solution | 0.966 | 同上 + 大規模アンサンブル |
| Dark-AGI-Biohub Cell Tracking Solution | 0.952 | 同上 |
| Biohub Cell Tracking（3日前更新） | 0.917 | 別系統のリンク設計 |
| Biohub M001 ens3 sm6 sim2 | 0.917 | 前処理多様性(tophat有無)、外見コスト付き割当(`sim2`) |
| **Biohub - Track Your Cells Development（本notebook）** | **0.916** | — |
| Biohub 0.902 Motion + Division Calibration | 0.901 | — |

**注意すべき文脈**: 0.95以上の上位群は、このVaultで2026-08-04に確認したとおり
**メトリックハック**（提出座標のボリューム範囲外・負の時刻を弾いていない指標実装の脆弱性を突く）を
含むものが大半。本notebookの0.916は**正攻法だけで到達した値**であり、0.969と単純比較すべきではない。

**関連文献**

- [Higher-Order Cell Tracking Transformer (arXiv 2607.11754)](https://arxiv.org/abs/2607.11754):
  **エッジ中心**のアーキテクチャで、深い事前学習画像エンコーダなしにCell Tracking Challengeで
  SOTAを達成。本notebookが「U-Netで検出 → 後処理で辺を直す」と2段構えにしている部分を、
  **辺そのものを主役にした1つのモデル**で解く方向。後処理の手作業が丸ごと不要になる可能性がある。
- [Cell Tracking according to Biological Needs — Multi-Hypothesis Tracker with Aleatoric Uncertainty (arXiv 2403.15011)](https://arxiv.org/html/2403.15011v3):
  1対1割当の代わりに**複数仮説を保持し、不確実性を明示的にモデル化**する。
  本notebookが「後処理で分裂を復元する」形で外付けしているものを、追跡アルゴリズムに内蔵する設計。

**改善提案**

1. **DeepCenter拒否権のしきい値を、ターゲット別に分ける**。
   現状は分裂修復もギャップ埋めも同じしきい値を使っている。
   ギャップ埋めの中間ノードは「両端が確実」という強い事前情報があるので、
   分裂修復より**緩いしきい値でよい**はず。修復の種類ごとにしきい値を切ると回収率が上がる余地がある。
2. **`GAP_REFINE_MAX_SHIFT_UM` による却下を「捨てる」から「縮める」に変える**。
   現状は補正量が大きすぎると補正を丸ごと放棄して中点に戻す。
   代わりに**上限までクリップして採用**すれば、7.0µm判定を通る確率は上がる。
   却下したケースの統計（`gap_refine_rejected_shift`）を出しているので、効果は即測定できる。
3. **ラインフィットの窓と重み(w=3, wt=0.85)を検証データでグリッドサーチする**。
   手動決め打ちの2つのつまみで、しかも**トポロジーを壊さない=安全**な改善なので、
   投入コストあたりの期待リターンが最も高い。
4. **8方向TTAへのモンキーパッチに、置換失敗の検出を入れる**。
   現状 `_s.replace(_old, _new)` は、上流の文字列が1文字でも変われば**静かに失敗**する。
   `if _old not in _s: raise RuntimeError(...)` を1行足すだけで、
   「TTAが効いていないまま高スコアだと思い込む」事故を防げる。
   このnotebookは設定ドリフト・ガードを実装するほど再現性に厳しいのに、ここだけ穴が空いている。
5. **エッジ中心Transformer（上記文献）の考え方を、既存の後処理の順序に取り入れる**。
   現状の後処理は「再リンク → ギャップ埋め → 分裂修復 → 短小除去 → 平滑化」の固定順序で、
   前段の判断を後段が覆せない。辺のスコアを一度全部並べ直してから
   まとめて決める（=複数仮説を後まで保持する）だけでも、順序依存の取りこぼしは減らせる。

---

## 2. Playground S6E8：合成データの格子構造を、ルックアップ表に変える

**原著者**: Koushik Kumar Dinda ([@koushikkumardinda](https://www.kaggle.com/koushikkumardinda))・44 votes（Bronze）
**リンク**: https://www.kaggle.com/code/koushikkumardinda/robust-ensembling-target-encoding-pipeline
**スコア**: Public 0.96464

### 学べる主要テクニック

- **「文字列化（Stringify）」トリック**。`sleep_hours` などの連続値を `.astype(str)` で
  カテゴリに変換し、ターゲットエンコーディングをかける。
  Playgroundのデータは小さな実データ（約7,500行）から生成した合成データで、
  **値が格子状に並ぶ**。木モデルは大小関係の分割しか作れないので格子構造を効率よく表現できないが、
  各格子点を「その値のときの陽性率」に変えてしまえば、**生成ルールの逆算**になる。
- **fold内でのみターゲットエンコーダを`fit`する**。
  `TargetEncoder(smoothing=10)` を学習foldだけで`fit`し、検証fold・testを`transform`。
  分割前に全データでエンコードすると**検証foldの正解が特徴量経由で漏れる**。
  CVだけ跳ね上がってLBで再現しない、という典型的な失敗を回避している。
- **ランク平均でスケールを揃える**。AUCは順位のみの指標なので、
  確率のキャリブレーションにコストをかけず `rank(pct=True)` で揃えてから混ぜる。
- **メタモデルを極端に正則化する**（`max_depth=2`, `num_leaves=3`, `min_child_samples=100`）。
  自由度を与えるとメタモデルはOOF予測を暗記してしまう。事実上の決定株に縛り、
  「ベースモデルの予測を微調整する」以上のことをさせない。
- **生の特徴量を2つだけメタ層に注入する**（`age`, `daily_screen_time_hours`）。
  「この年齢帯ではベースモデルが系統的に外している」という**条件つき補正**を学ばせる狙い。

### 評価指標の要約

**ROC-AUC**。全ての「陽性1件・陰性1件」ペアのうち、陽性に高いスコアを付けられた割合。
**順位だけで決まり、絶対値に依存しない**。しきい値を1つに固定しなくてよいので、
「モデルの良し悪し」と「しきい値選びの巧拙」が混ざらない。
本notebookのランク平均は、この性質を直接利用した設計。

### 改善点の考察

**他notebookとの比較**（`?searchQuery=s6e8` で20件を確認）

| notebook | スコア | 本notebookとの差 |
|---|---|---|
| S6E8 Addiction Blend LB 0.97117 / Elite Rank Average | 0.97117 | 公開提出のランク平均のみ（学習内容は薄い） |
| Why Every S6E8 Notebook Above 0.97110 Overfits | 0.97115 | 昨日レビュー。上位帯は選択バイアスと主張 |
| S6E8 Addiction LB 0.97113 | 0.97113 | 47モデルのOOFライブラリ + 加重ブレンド |
| S6E8: LGBM \| LB 0.96965 | 0.96965 | **単体LGBM。本notebookより高い** |
| S6E8: HistGradientBoosting | 0.96945 | 単体 |
| S6E8: CatBoost | 0.96813 | 単体。**CatBoostを使っていない** |
| **📱 Robust Ensembling & Target Encoding（本notebook）** | **0.96464** | — |

**ここが今日一番の発見**: 本notebookは2モデルのスタッキングをしているのに、
**単体のLGBM（0.96965）にすら 0.005 負けています**。
スタッキングは普通ベースモデルを下回らないので、これは異常です。
実際にコードを読むと、原因の有力候補が見つかりました。

```python
meta_X_train = pd.DataFrame({'lgbm_oof_rank': ..., 'xgb_oof_rank': ...})
meta_X_test  = pd.DataFrame({'lgbm_test_rank': ..., 'xgb_test_rank': ...})
```

**学習側とテスト側で列名が違います。** LightGBMは特徴量名を保持するため、
この状態は少なくとも「同じ意味の列が同じ名前で渡っていない」ことを意味し、
警告か例外、あるいは列の対応ずれを招きます。
高スコアnotebookを眺めるより、**この1個のバグを追うほうが学習効果が高い**と判断しました。

**関連文献・技術記事**

- [The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data (NVIDIA)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/):
  スタッキングは**2層を超えると利得が急速に減り、各層で過学習リスクが複利で効く**ため、
  実務ではメタ層1段で十分という指摘。またメタ学習器は**単純なほど良い**（本notebookの方針は正しい）。
- [Target Encoding + Noise Injection (Kaggle Discussion)](https://www.kaggle.com/discussions/general/579733):
  ターゲットエンコーディングに**微小なノイズを注入**して過学習を抑える手法。
  平滑化パラメータは5〜10が既定として妥当（本notebookの`smoothing=10`は妥当な範囲）。

**改善提案**

1. **列名の不一致を直す**（最優先・おそらく数点分）。
   `assert list(meta_X_train.columns) == list(meta_X_test.columns)` を1行入れるだけで検出できる。
   本質的には、**メタ特徴量を train/test で同じ関数から作る**べき。
2. **メタモデルを使わず、ランク平均だけで比較する**。
   スタッキングが単体LGBMを下回っている以上、まず
   `0.5 * rank(lgbm) + 0.5 * rank(xgb)` の素朴なブレンドをベースラインとして測るべき。
   **メタモデルが本当に価値を足しているかを確かめてから**複雑にする。
3. **CatBoostを3本目のベースモデルに加える**。
   CatBoostは順序付きターゲット統計を**内部で**持っており、
   外部の`TargetEncoder`とは違う癖の予測を出す。単体で0.96813が出ている実績もあり、
   **間違え方の多様性**という意味で最も費用対効果が高い追加候補。
4. **早期終了の検証セットを、OOFの検証foldと分ける**。
   現状は木の本数を検証foldで選び、同じfoldでOOFスコアを測っているので、
   OOFがわずかに楽観的になる。内側にもう一段CVを切るか、学習foldから小さく切り出す。
5. **「文字列化」の対象列を、ユニーク値数で自動選択する**。
   現状3列を手で指定しているが、`df[col].nunique() < 閾値` で機械的に選べば
   見落としがなくなる。合成データでは他の列にも格子構造が潜んでいる可能性がある。

---

## 3. RSNA Knee Abnormality Detection：勝負はモデルの手前で決まっている

**原著者**: ANHAD MAHAJAN ([@anhadmahajan06](https://www.kaggle.com/anhadmahajan06))・65 votes
**リンク**: https://www.kaggle.com/code/anhadmahajan06/rsna-knee-take-care-of-your-knee
**スコア**: Public 0.914（notebook内では3段構成で0.920到達を主張）

### 学べる主要テクニック

- **ターゲット別のTTAプーリング戦略**。10個のスライディング窓の予測を、
  所見の性質ごとに違う方法で集約する。
  `max`＝骨折・骨挫傷・半月板・ベーカー嚢腫（**限局病変**。平均すると信号が1/10に薄まる）、
  `top2`＝ACL・MCL（**細長い靭帯**）、`mean`＝変形性関節症・関節液貯留・滑膜炎（**びまん性**）。
  マクロ平均AUCは各ターゲットが独立に効くので、集約方法を所見ごとに変えるのが素直に正解になる。
- **多言語レポートの正規表現解析による弱教師ラベル生成**。
  英・独・仏・西・蘭・土・希・露・ブルガリア語の同義語を辞書化し、
  **語幹と修飾語の距離（前後55文字）**で「内側半月板の断裂」と「外側半月板は正常」を区別する。
  出力は0/1ではなく**確信度つきの軟らかいラベル**（手動確認=3.0、レポート由来=0.25〜1.0）。
  「テキスト由来のラベルは人手ラベルほど信用できない」を**重みで明示的にモデルに伝える**。
- **スライス順序の幾何学的復元**。DICOMのファイル名やUIDは空間順序と無相関（Spearman≈0.009）。
  面の法線 `n = r_x × r_y` に原点 `p` を射影した `k = p·n` でソートして初めて、
  医学的に意味のあるボリュームになる。
- **解剖学的事前分布をアテンションのバイアスに焼き込む**（`SlotHead`）。
  「ACLを見るなら主に矢状断」を `SLOT_PRIOR_TABLE` として
  アテンションスコアに**加算**する。ハード制約ではないので、データが強く反対すれば上書きできる。
- **モデルの指紋照合**。固定シードの乱数画像を通した出力を指紋として記録し、
  重み読み込み後に許容誤差0.002で照合。24メンバーのアンサンブルで
  「重みAのつもりが実はB」という**静かな取り違え**を検出する。
- **デュアルGPU作業キュー＋途中保存**。メンバーを1つ処理するたびに提出ファイルを更新するので、
  タイムアウトしても**そこまでのアンサンブルは提出される**。
- **ターゲット別の融合ゲート（α）**。Stage 3でベーカー嚢腫と骨折だけ **α=0.0**（融合しない）。
  DINOv2が`max`で捉えた鋭い信号を、別モデルの平均で鈍らせないため。
- **多段フォールバック**。最上位で `except Exception` を捕まえ、
  全ターゲット0.5の提出を必ず書き出す。ただし `LabelSourceError`（設定ミス）だけは再送出して止める。
  **握り潰してよい失敗と、気づくべき失敗を区別している**のが上手い。

### 評価指標の要約

**12ターゲットのAUCのマクロ平均**（各所見で個別にAUCを計算して単純平均）。
12の所見は出現頻度が大きく違うため、まとめて1つのAUCにすると多数派の所見だけで点が決まる。
マクロ平均は**すべてのターゲットを等しく重み付け**するので、
「珍しいが臨床的に重大な所見（骨折など）」を無視できない。
本notebookのターゲット別プーリング／ターゲット別αは、この性質に**直接最適化した設計**。

### 改善点の考察

**他notebookとの比較**（`?searchQuery=rsna+knee` で20件を確認）

| notebook | スコア | 本notebookが採用していない要素 |
|---|---|---|
| Bend the Knee to DinoV3 (ensembled) | 0.922 | **DINOv3**（本notebookはv2止まり） |
| RSNA Knee Abnormality / DINOsaur V3 | 0.922 | より新しいバックボーン＋大規模アンサンブル |
| RSNA Knee DINO-RadImageNet Rank Ensemble | 0.920 | ランク融合の重み最適化 |
| RSNA Knee frontier v48 | 0.917 | — |
| **RSNA Knee: Take Care Of Your Knee（本notebook）** | **0.914** | — |
| RSNA Knee baseline v1 | 0.891 | 388 votes。最も広く読まれている土台 |

**関連文献**

- **OrthoFoundation**（[RSNA 2026 AI Challenge の関連研究](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri)）:
  膝X線・MRIの**未ラベル120万枚**を DINOv3 バックボーンで自己教師あり事前学習した
  筋骨格特化の基盤モデル。14の下流タスクでSOTA。
  本notebookが DINOv2（自然画像）+ RadImageNet（放射線画像一般）を混ぜて作ろうとしている
  「ドメイン適応した表現」を、**事前学習の段階で作ってしまう**方向。
- [MM-DINOv2: Adapting Foundation Models for Multi-Modal Medical Image Analysis (arXiv 2509.06617)](https://arxiv.org/pdf/2509.06617):
  複数モダリティ（＝本コンペでいう6スロット）を扱うために DINOv2 を適応させる手法。
  **欠損モダリティへの頑健性**を明示的に扱っており、本notebookの
  `stochastic_slot_mask`（学習時のスロットドロップアウト）と同じ問題意識に、より体系的に答えている。

**改善提案**

1. **バックボーンを DINOv3 に差し替える**。上位3本（0.922）が揃って DINOv3 系で、
   本notebookだけ v2 に留まっている。差の 0.008 の大部分はここで説明できる可能性が高い。
   `build_model` はHuggingFaceの`AutoModel`を使っているので、差し替えコストは小さい。
2. **Stage 2 のブレンド重み W=0.45 を、ターゲット別にする**。
   Stage 3 ではターゲット別のαを使っているのに、Stage 2 だけ12ターゲット一律。
   **思想が一貫していない**。マクロ平均AUCなら、各ターゲットで独立に最適な重みを選べる。
3. **Stage 2 のスライス順序を Stage 1 と揃える**。
   Stage 1 は `ImagePositionPatient` の法線射影で厳密に並べているのに、
   Stage 2 は `InstanceNumber` 順。同じnotebook内で精度の違う前処理が同居しており、
   Stage 2 の予測品質を下げている可能性がある。
4. **ターゲット別のαと、Stage 2 の重みを、OOFで最適化する**。
   現状の 0.50 / 0.15 / 0.0 / 0.45 はすべて手動決め打ち。
   `macro_auc` を自前実装しているのだから、**Optunaで12次元の重みを制約付き最適化**できる。
   ただし過学習を避けるため、重みは非負・和が1に制約すること。
5. **レポート解析の再現率を、言語別に測って報告する**。
   現状、多言語辞書の網羅性を評価する仕組みがない。
   16施設・十数言語のデータなので、**特定言語だけラベル生成が壊れている**可能性は十分ある。
   言語別に「所見が1つも抽出できなかったレポートの割合」を出すだけで、抜けが見える。

---

## 今日のまとめ

3本とも、**モデルの中身より外側で差が付く**という同じ結論に着地しました。

- Biohubは、後処理で「増やす」たびに**別モデルの拒否権で「減らす」**。
- RSNAは、正規表現とDICOM幾何という**地味な工程**が精度の土台を作り、
  最後の融合重みを**ターゲット別に分ける**ことで指標に直接効かせる。
- Playgroundは、合成データという**データ自体の性質**を突いた前処理が本体で、
  そのうえで**メタ特徴量の列名という凡ミス**でスタッキングの利得を失っている。

特に3本目は「スタッキングしたのに単体モデルに負けている」という結果から
バグに辿り着けたケースで、**スコアが期待より低いときは、まず実装を疑う**という
当たり前のことを改めて確認できました。高スコアnotebookを眺めるだけでは、この経験は得られません。
