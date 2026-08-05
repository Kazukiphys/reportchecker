# レポート一致チェックアプリ（Streamlit版）

高校の授業で提出されたPDFレポート同士を比較し，一致率の高い組み合わせと一致候補箇所を確認するWebアプリです。

GitHubに公開して動かす場合は，GitHub Pagesではなく Streamlit Community Cloud を使うのが前提です。

Streamlit Cloud 用の設定は [.streamlit/config.toml](.streamlit/config.toml) に入れています。

## できること

- 複数PDF（目安30〜40件）を一括アップロード
- 全ペア比較で一致率を算出
- 一致率の高い順に一覧表示
- OCRフォールバックで画像PDF（スキャンPDF）も自動的に比較対象化
  - 詳細ビューで左右にPDFを表示し，一致候補文をハイライト表示
- 結果をCSVでダウンロード

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 実行

```bash
streamlit run app.py
```

ブラウザで表示されたURLを開いてください。

## GitHubで公開して動かす方法

1. このリポジトリをGitHubにpushします。
2. Streamlit Community Cloud で New app を選びます。
3. Repository にこのGitHubリポジトリを指定します。
4. Main file path に `app.py` を指定します。
5. Deploy を実行します。

この構成では、公開時の本体は `app.py` です。`main.py` と `templates/` は FastAPI版の別実装として残していますが、GitHub公開で動かす対象は Streamlit版です。

## 仕組み（簡易）

- PDFテキスト抽出: `pypdf`
- OCR（必要時のみ）: `pypdfium2` + `rapidocr-onnxruntime`（未導入時は自動フォールバック）
- レポート全体の一致率: 文字n-gramのTF-IDF + コサイン類似度
- 一致箇所候補:
  - 同一文候補: 文の正規化後に一致する文を抽出
  - 類似文候補: 文同士のTF-IDF類似度で候補抽出
- ハイライト表示: `PyMuPDF` で一致候補文をPDF上に注釈

## 補足

FastAPI版の画面や `/analysis/{analysis_id}` のようなURLは、`main.py` 側の別実装です。GitHub公開でそのまま動かす用途では、`app.py` の Streamlit 版を使ってください。

## 注意事項

- 判定は補助目的です。最終判断は教員が行ってください。
- OCR結果は誤認識を含む可能性があります。疑わしい箇所は原本PDFで確認してください。
- 個人情報を含むPDFは公開クラウド環境ではなく，ローカルまたは校内環境で運用してください。
