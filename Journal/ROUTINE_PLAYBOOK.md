# 毎朝のKaggle解説ルーティン 実行手順書

このファイルは、スケジュールタスク（毎朝7:00/7:30/8:00/8:30の4回チェック）が読み込んで実行する詳細手順です。
このドキュメントだけで、今回の会話の文脈なしに実行できるように書いています。

## 目的

Kaggleの上位入賞者・上位得票のコード（Playground 1本 + 実コンペ1本、計2本/日）を、
Python初心者にも「何をしているか」「なぜそうするのか」が分かるように解説付きでipynb化し、
GitHubリポジトリ https://github.com/tsukasahamaoka9-source/DS_Practice にコミットする。

目的は模写ではなく学習：手法・作法を学び、日々の前進を履歴として残すこと。

## 前提・パス

- 作業フォルダ（Read/Write/Edit/Grep/Glob用）: `/Users/tsukasahamaoka/Document/DS_Practice/Journal`
- 同フォルダ（bash用マウントパス）: `/sessions/keen-admiring-davinci/mnt/Journal`（※セッションIDは変わる場合があるため、bash内で `ls` 等により実際のマウントパスを都度確認すること。分からなければ `find / -maxdepth 4 -iname "Journal" 2>/dev/null` で探す）
- GitHubリポジトリ: owner=`tsukasahamaoka9-source`, repo=`DS_Practice`, branch=`main`
- 状態管理ファイル: `Journal/.routine_status/{YYYY-MM-DD}.json`
- 成果物の一時置き場（ローカル）: `Journal/Daily_Kaggle_Review/{YYYY-MM-DD}/`
- **重要**: `Journal/kaggle.json` と `Journal/.github_token` は絶対にGitHubにアップロードしない。これらはこのフォルダ直下にあるだけで、今回のフロー（ブラウザ経由でのアップロード）では使わない。

## ネットワーク制約（超重要）

このサンドボックスのbash/シェルは外部ネットワーク（github.com, kaggle.com, pypi.org等）に一切接続できない（プロキシで403ブロックされる）。
そのため:

- `git clone` / `git push` / `pip install` はすべて失敗する。**試みないこと。**
- Kaggleの閲覧・コード取得、GitHubへのコミットは、必ず **Control_Chrome（またはclaude-in-chrome）MCP経由でユーザーの実ブラウザを操作して** 行う。
- ipynbファイルの組み立てはPython標準の`json`モジュールで行う（`nbformat`ライブラリはインストールできないため使わない）。

## 実行フロー（各チェック時刻ごとに必ず最初に行うこと）

1. 現在のローカル日付（YYYY-MM-DD）を確認する。
2. `Journal/.routine_status/{today}.json` を読む。存在しない場合は `{"status": "pending"}` とみなす。
3. `status` が `"done"` または `"notified"` なら、**即座に何もせず終了**する（その日はもう処理済み）。
4. `status` が `"pending"` の場合のみ、以下へ進む。

### ブラウザ疎通チェック

5. Control_Chrome（またはclaude-in-chrome）のツールをToolSearchでロードし、`list_tabs`（または同等の軽量な呼び出し）を試す。
   - エラー（"Chrome is not running" 等）が返る、またはタイムアウトする場合 → **PC/ブラウザ不在**と判断。
     - 現在時刻が **8:30より前** なら、状態ファイルを更新せず、ユーザーに通知もせず、静かに終了する（次のチェックに委ねる）。
     - 現在時刻が **8:30以降**（最終チェック）なら、`Journal/.routine_status/{today}.json` に `{"status": "notified"}` と書き込み、ユーザーに短いメッセージで「今朝はPCが見つからなかったため、今日のKaggleレポートはスキップしました」と伝えて終了する。
   - 成功する場合 → **PC稼働中**と判断し、下記「本編：日次ルーティン」に進む。

## 本編：日次ルーティン（ブラウザ疎通確認できた場合のみ実行）

### A. 重複回避のための履歴確認

- `Journal/Daily_Kaggle_Review/` 配下の既存の日付フォルダ名とその中のファイル名（notebookのslug）を確認し、過去に扱ったコンペ・notebookのリストを把握する。同じコンペ・同じnotebookは避ける（同じコンペを再訪するのは可だが、その場合は別のnotebookを選ぶ）。

### B. コンペ選定

**1本目: Playground Series**
- Kaggleの Playground Series（`https://www.kaggle.com/competitions?hostSegmentIdFilter=5&sortOption=recentlyCreated` 等、またはCompetitionsページで"Playground"タグ・"Getting Started"カテゴリから検索）の中から、現在進行中または直近の回を選ぶ。既出のものは避ける。

**2本目: 実際のコンペ**
- 対象は「現在開催中、または最近終了して2週間以内でデータ・コードタブに引き続きアクセスできるもの」。
- 参加者数・データ量が多いものを優先。
- 分野は生物・医療・ヘルスケア・金融・商業/ビジネス課題を優先。土木・地質など、専門から離れる分野は避ける（ユーザーの明示的な希望）。
- Code Competition（隠しテストセット・提出がnotebook経由のみ）で、終了後にコードタブが非公開になっているものは避ける。

両方とも、コンペの `Code` タブを開き、`Most Votes` でソートし、上位（できれば金メダル相当・投票数が多い）の公開notebookを1つ選ぶ。

### C. notebookの取得と解説付きipynbの作成

各notebookについて:

1. Control_Chrome（JS未実行で内容が取れない場合はclaude-in-chromeの`get_page_text`にフォールバック）でnotebookページを開き、コードセルとMarkdownセル（著者の説明文）の内容を可能な限り読み取る。
2. 取得した内容をもとに、以下の構成でipynb（nbformat v4のJSON構造）を組み立てる。**Pythonの`json`モジュールで直接JSON文字列を組み立てて`.ipynb`として保存すればよい**（nbformatライブラリ不要）。

   最小構造:
   ```json
   {
     "cells": [ {"cell_type": "markdown", "metadata": {}, "source": ["..."]},
                {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["..."]} ],
     "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}},
     "nbformat": 4,
     "nbformat_minor": 5
   }
   ```

3. 構成ルール:
   - 先頭に見出しMarkdownセル: コンペ名、notebookタイトル、原著者名、元notebookへのリンク、手法の概要（1段落）、「これは学習目的の解説付き写しであり、未実行（コード自体は変更していないが出力は含まない）」旨の断り書き。
   - 元のコードセルはほぼそのまま保持する（大きな改変はしない）。
   - **各コードセルの直前に、独立したMarkdownセルを1つ挿入**し、「何をしているか（What）」と「なぜそうするのか（Why）」を簡潔な日本語で書く。行内コメントは最小限に留め、説明の大部分はMarkdownセルに書く（読みやすさのため）。
   - Python初心者が読んでも分かるよう、専門用語には一言補足を添える。

4. ファイル名: `Journal/Daily_Kaggle_Review/{today}/playground_{slug}.ipynb`、`Journal/Daily_Kaggle_Review/{today}/competition_{slug}.ipynb` として保存する。

### D. 日次README作成

`Journal/Daily_Kaggle_Review/{today}/README.md` を作成し、以下を簡潔に日本語でまとめる:
- 今日扱った2つのnotebook（コンペ名、原著者、リンク）
- それぞれで学べる主要テクニック・手法（箇条書きで3〜5点ずつ）
- 一言まとめ

### E. GitHubへのコミット（ブラウザ経由）

bashからのgit pushは使えないため、必ずブラウザで行う。

1. `https://github.com/tsukasahamaoka9-source/DS_Practice/upload/main/Daily_Kaggle_Review/{today}` をControl_Chrome/claude-in-chromeで開く。
2. ファイルアップロードのツール（`mcp__claude-in-chrome__file_upload` 等、deferredならToolSearchでロード）を使い、**Cで作成した2つの.ipynbファイルとDで作成したREADME.mdの、合計3ファイルだけ**をローカルの`Journal/Daily_Kaggle_Review/{today}/`から添付する。フォルダ内の他のファイル（kaggle.json、.github_token等）は絶対に含めない。
3. コミットメッセージ欄に `Add {today} Kaggle review: playground_{slug} + competition_{slug}` のように入力する。
4. 「Commit changes」ボタンをクリックしてmainブランチに直接コミットする。
5. 可能であれば `Daily_Kaggle_Review/README.md`（インデックスページ）も同様の方法で更新し、今日の日付・扱ったコンペ名・リンクを1行追記する（既存ファイルの編集は `https://github.com/tsukasahamaoka9-source/DS_Practice/edit/main/Daily_Kaggle_Review/README.md` から行う）。

### F. 状態更新と通知

- `Journal/.routine_status/{today}.json` に `{"status": "done", "completed_at": "<ISO時刻>", "playground": "{slug}", "competition": "{slug}"}` を書き込む。
- ユーザーに短いチャットメッセージで完了報告する（2本のタイトル・リンク・一言学びポイント程度。長々と説明しない）。

## 品質・安全に関する注意

- 生成するのは「学習用の解説付き写し」であり、原著者のコードを尊重し、リンクと著者名を必ず明記する。
- 分野選定は生物・医療・金融・商業系を優先し、土木・地質等の専門外分野は避ける。
- 同じコンペ・notebookの重複は避ける（Aの履歴確認を必ず行う）。
- kaggle.json・.github_tokenは今回のフローでは使用しない。誤ってGitHubにアップロードしないこと。
- 何か想定外のエラー（ブラウザ操作失敗、notebookが読み取れない等）が起きた場合は、無理に処理を続けず、状態ファイルは`"pending"`のまま残し、ユーザーに簡潔にエラー内容を報告する（次回チェックで再試行できるようにするため、`"done"`や`"notified"`にはしない）。
