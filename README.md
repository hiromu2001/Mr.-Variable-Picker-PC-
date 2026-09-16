# Mr. Variable Picker PC

PC上で動作する、カメラ映像を利用したリアルタイム人物・属性分析システムです。

Raspberry Pi版の Mr. Variable Picker をPC向けに発展させ、DeepFaceを利用して顔画像から年齢・性別・表情を推定します。

## 特徴

- Webカメラだけで動作
- OpenCVに同梱されているHaar Cascadeを使用するため、顔検出モデルを別途配置する必要なし
- DeepFaceの学習済みモデルは初回起動時に自動ダウンロード
- 人物をID単位で追跡
- 年齢・性別・表情を複数回推定して安定化
- 人物ごとの滞在時間をCSVへ記録

## 処理フロー

~~~text
カメラ映像
   ↓
OpenCV Haar Cascade
   ↓
Centroid Tracker
   ↓
人物IDを付与
   ↓
DeepFace
   ├─ 年齢
   ├─ 性別
   └─ 表情
   ↓
複数回の推定結果を蓄積
   ↓
代表値を算出
   ↓
滞在時間を計算
   ↓
CSVへ保存
~~~

## 必要環境

- Python 3.9〜3.11程度を推奨
- Webカメラ
- Windows / macOS / Linux

## 最短セットアップ

### 1. クローン

~~~bash
git clone https://github.com/hiromu2001/Mr.-Variable-Picker-PC-.git
cd Mr.-Variable-Picker-PC-
~~~

### 2. ライブラリをインストール

~~~bash
pip install -r requirements.txt
~~~

### 3. 起動

~~~bash
python main.py
~~~

**これだけで起動できます。**

追加の顔検出モデルを手動でダウンロードしたり、modelsフォルダへファイルを配置したりする必要はありません。

## 初回起動について

初回起動時にはDeepFaceが必要な学習済みモデルを自動的にダウンロードします。

~~~text
python main.py
       ↓
DeepFaceがモデルを確認
       ↓
未取得なら自動ダウンロード
       ↓
ローカルへキャッシュ
       ↓
顔分析開始
~~~

初回だけモデルのダウンロードに時間がかかる場合があります。

2回目以降はローカルにキャッシュされたモデルが利用されます。

## 出力

人物の追跡が終了すると、logs/ にCSVが保存されます。

~~~text
logs/
└── analytics_YYYYMMDD_HHMM.csv
~~~

出力項目：

| 項目 | 内容 |
|---|---|
| end_timestamp | 人物の追跡終了時刻 |
| gender | 推定された性別 |
| age_stable | 推定年齢の中央値 |
| top_expression | 最頻の表情 |
| result | stay / pass |
| total_dwell_sec | 滞在時間（秒） |

## 技術的な工夫

### 人物追跡

OpenCVで顔を検出した後、顔の中心座標を利用したCentroid Trackerで人物を追跡します。

一時的に顔が検出できなくなっても、最大30フレームまでは同じIDを保持します。

### DeepFaceの推論頻度を制御

DeepFaceは毎フレーム実行すると計算負荷が大きいため、人物IDとフレーム番号を利用して、人物ごとに5フレームに1回の頻度で推論します。

### 推定結果の安定化

1回のAI推論結果をそのまま利用せず、人物ごとに複数回の推定結果を蓄積します。

- 年齢：中央値
- 性別：最頻値
- 表情：最頻値

これにより、単一フレームの推定結果の揺れを分析結果にそのまま反映しない構成にしています。

### 滞在判定

人物の分析データから滞在時間を算出します。

~~~text
滞在時間 >= 2秒 → stay
滞在時間 <  2秒 → pass
~~~

閾値は analytics.py の DWELL_THRESHOLD で変更できます。

## ファイル構成

~~~text
Mr.-Variable-Picker-PC-/
├── main.py
├── tracker.py
├── analytics.py
├── requirements.txt
└── logs/
~~~

### main.py

カメラ入力、顔検出、人物追跡、DeepFace推論、画面表示を制御します。

### tracker.py

Centroid Trackerによる人物IDの管理を行います。

### analytics.py

人物ごとの推定結果を蓄積し、代表値・滞在時間を計算してCSVへ出力します。

### requirements.txt

Pythonの依存ライブラリを定義しています。

## 使用技術

| 技術 | 用途 |
|---|---|
| Python | アプリケーション実装 |
| OpenCV | カメラ入力・顔検出・映像処理 |
| DeepFace | 年齢・性別・表情推定 |
| NumPy | 数値計算・属性の集約 |
| SciPy | 人物追跡の距離計算 |
| CSV | 分析結果の保存 |

## モデルファイルについて

このリポジトリにはDeepFaceの学習済みモデルを含めていません。

これは意図した構成です。

モデルをGitHubへ直接配置するのではなく、DeepFaceにモデル管理を任せることで、

- Gitリポジトリの肥大化を防ぐ
- モデルファイルを手動配置する必要がない
- クローン後のセットアップを簡単にする

というメリットがあります。

また、OpenCVのHaar Cascadeについても cv2.data.haarcascades からOpenCV同梱のファイルを参照するため、別途モデルファイルを用意する必要がありません。

## 今後の改善案

- YOLOなどの物体検出モデルへの変更
- ByteTrack / DeepSORTなどによる追跡精度向上
- GPUを利用したDeepFace推論の高速化
- 推論頻度の動的制御
- SQLite / PostgreSQLなどへの保存
- 時間帯・曜日別の分析
- 属性と滞在時間のクロス集計
- ダッシュボードによる可視化
- 複数カメラへの対応
- 匿名化・プライバシー保護を考慮した運用

## 注意事項

本プロジェクトは技術検証を目的としたプロトタイプです。

年齢・性別・表情などの推定結果はAIによる推定値であり、実際の人物属性を保証するものではありません。

実店舗などで利用する場合は、撮影・保存・分析に関する法令、社内規程、利用者への説明、プライバシーへの配慮などを確認してください。

また、DeepFaceおよび各モデルのライセンス・利用条件についても、実運用前に確認してください。

## License

ライセンスについては、必要に応じて追加してください。
