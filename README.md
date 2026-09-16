# Mr. Variable Picker PC

PC上で動作する、カメラ映像を利用したリアルタイム人物・属性分析システムです。

Raspberry Pi版の Mr. Variable Picker をPC向けに発展させ、DeepFaceを利用して顔画像から年齢・性別・表情を推定します。

> **特徴：モデルファイルをリポジトリへ直接含める必要はありません。**
> DeepFaceが初回起動時に必要な学習済みモデルを自動ダウンロードします。

## 概要

カメラ映像から顔を検出し、人物ごとにIDを付与して追跡します。

追跡した人物に対して一定間隔でDeepFaceによる属性推定を行い、複数回の推定結果を蓄積します。人物が画面から離れた時点で、それまでの結果を集約してCSVへ保存します。

~~~text
カメラ映像
   ↓
顔検出（OpenCV Haar Cascade）
   ↓
人物追跡（Centroid Tracker）
   ↓
DeepFaceによる属性推定
   ├─ 年齢
   ├─ 性別
   └─ 表情
   ↓
複数フレームの推定結果を蓄積
   ↓
人物単位で属性を安定化
   ↓
滞在時間を計算
   ↓
CSVへ出力
~~~

## 主な機能

- Webカメラからのリアルタイム映像取得
- OpenCVによる顔検出
- Centroid Trackerによる人物ID追跡
- DeepFaceによる年齢推定
- DeepFaceによる性別推定
- DeepFaceによる表情推定
- 複数フレームの推定結果を利用した属性の安定化
  - 年齢：中央値
  - 性別：最頻値
  - 表情：最頻値
- 人物ごとの滞在時間計測
- 滞在時間2秒以上を stay、2秒未満を pass として分類
- 分析結果のCSV保存

## Raspberry Pi版との違い

このPC版では、属性推定に **OpenVINOの直接推論ではなくDeepFace** を使用しています。

DeepFaceを利用することで、顔画像から年齢・性別・表情などをまとめて推定できます。

また、DeepFaceのモデル管理機能により、モデルファイルをGitリポジトリへ直接コミットせず、初回起動時に必要なモデルを取得する構成にしています。

## 使用技術

| 技術 | 用途 |
|---|---|
| Python | アプリケーション実装 |
| OpenCV | カメラ入力・顔検出・映像処理 |
| DeepFace | 年齢・性別・表情推定 |
| NumPy | 数値計算・属性の安定化 |
| SciPy | 人物追跡の距離計算 |
| CSV | 分析結果の保存 |

## ディレクトリ構成

~~~text
Mr.-Variable-Picker-PC-/
├── main.py
├── tracker.py
├── analytics.py
├── models/
│   └── haarcascade_frontalface_default.xml
└── logs/
    └── analytics_YYYYMMDD_HHMM.csv
~~~

※ DeepFaceが使用する学習済みモデルは通常リポジトリ内には配置しません。初回実行時にDeepFace側でダウンロードされます。

## 各ファイルの役割

### main.py

アプリケーションのエントリーポイントです。

カメラからフレームを取得し、

1. 顔検出
2. 人物追跡
3. DeepFaceによる属性推定
4. 属性の安定化
5. 映像への結果表示
6. CSVへの記録

までを制御します。

DeepFaceによる推論は毎フレームではなく、人物IDとフレームカウンターを利用して一定間隔で実行することで、計算量を抑えています。

### tracker.py

Centroid Trackerを実装しています。

顔検出によって得られた矩形の中心座標を計算し、前フレームとのユークリッド距離を利用して人物IDを維持します。

最大30フレームまで人物が一時的に検出されなくてもIDを保持します。

### analytics.py

人物ごとの属性推定結果を蓄積・集約します。

追跡中の複数回の推定結果から代表値を計算し、人物の追跡終了時にCSVへ出力します。

## 属性の安定化

リアルタイム映像では、同じ人物を分析していてもフレームごとに推定値が変化することがあります。

そこで、単一フレームの結果をそのまま利用せず、複数回の推定結果を蓄積します。

### 年齢

外れ値の影響を抑えるため、中央値を使用します。

~~~text
age = median(推定された年齢)
~~~

### 性別

最も多く推定された性別を代表値として使用します。

~~~text
gender = mode(推定された性別)
~~~

### 表情

最も多く推定された表情を代表値として使用します。

~~~text
expression = mode(推定された表情)
~~~

このように、**単発のAI推論結果をそのまま分析データとして扱わず、時系列データとして集約する**設計にしています。

## 滞在判定

人物が最初に分析された時刻と最後に分析された時刻から滞在時間を計算します。

~~~text
滞在時間 >= 2秒 → stay
滞在時間 <  2秒 → pass
~~~

閾値は analytics.py の DWELL_THRESHOLD で変更できます。

## 出力データ

logs/ にCSVファイルが作成されます。

| 項目 | 内容 |
|---|---|
| end_timestamp | 人物の追跡終了時刻 |
| gender | 推定された性別 |
| age_stable | 推定年齢の中央値 |
| top_expression | 最頻の表情 |
| result | stay / pass |
| total_dwell_sec | 滞在時間（秒） |

例：

~~~csv
end_timestamp,gender,age_stable,top_expression,result,total_dwell_sec
2026-01-27 14:30:12,Male,34,neutral,stay,4.82
2026-01-27 14:30:18,Female,27,happy,pass,0.93
~~~

## セットアップ

### 1. リポジトリを取得

~~~bash
git clone https://github.com/hiromu2001/Mr.-Variable-Picker-PC-.git
cd Mr.-Variable-Picker-PC-
~~~

### 2. Python環境を用意

Python 3.9〜3.11程度の環境を推奨します。

仮想環境を利用する場合：

~~~bash
python -m venv .venv
~~~

Windows：

~~~bash
.venv\Scripts\activate
~~~

macOS / Linux：

~~~bash
source .venv/bin/activate
~~~

### 3. 必要ライブラリをインストール

~~~bash
pip install opencv-python deepface numpy scipy
~~~

### 4. モデルについて

**DeepFaceのモデルをGitHubへアップロードする必要はありません。**

DeepFaceは初回の分析実行時に必要なモデルを自動的にダウンロードします。

そのため、初回起動時は通常より時間がかかります。

~~~text
初回起動
   ↓
DeepFaceがモデルを確認
   ↓
未取得なら自動ダウンロード
   ↓
モデルをローカルキャッシュ
   ↓
顔分析開始
~~~

以降はキャッシュされたモデルを利用します。

### 5. Haar Cascadeについて

現在のコードでは models/haarcascade_frontalface_default.xml を参照しています。

このファイルがリポジトリに存在しない場合は、OpenCVに同梱されているHaar Cascadeを利用する方法もあります。

~~~python
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
~~~

この方式なら、Haar CascadeのXMLファイルをリポジトリへ含める必要もありません。

## 実行

カメラを接続した状態で、

~~~bash
python main.py
~~~

を実行します。

カメラ映像に人物ID・年齢・性別・表情が表示されます。

終了する場合は映像ウィンドウ上で q キーを押してください。

## 技術的なポイント

### 1. AI推論と人物追跡を分離

顔検出と属性推定を毎フレーム独立して行うのではなく、

~~~text
Face Detection
      ↓
Tracking
      ↓
Person ID
      ↓
Periodic AI Inference
      ↓
Time-series Aggregation
~~~

という構造にしています。

これにより、同一人物について複数回得られた推定結果を利用できます。

### 2. 推論結果を時系列データとして扱う

AIの1回の予測結果だけを見るのではなく、

~~~text
Person ID = 12

t1 → age 31 / Male / neutral
t2 → age 34 / Male / neutral
t3 → age 32 / Male / happy
t4 → age 35 / Male / neutral
...
~~~

のように蓄積し、最後に代表値を計算します。

これはリアルタイム画像認識を、そのまま分析データへ変換するための処理です。

## 今後の改善案

- YOLOなどの物体検出モデルへの変更
- ByteTrack / DeepSORTなどによる追跡精度向上
- GPUを利用したDeepFace推論の高速化
- 推論頻度の動的制御
- SQLite / PostgreSQLなどへの保存
- 時間帯・曜日別の分析
- 属性と滞在時間のクロス集計
- Streamlitなどを利用した分析ダッシュボード
- 複数カメラへの対応
- 匿名化・プライバシー保護を考慮した運用

## 注意事項

本プロジェクトは技術検証を目的としたプロトタイプです。

年齢・性別・表情などの推定結果はAIによる推定値であり、実際の人物属性を保証するものではありません。

実店舗などで利用する場合は、撮影・保存・分析に関する法令、社内規程、利用者への説明、プライバシーへの配慮などを確認してください。

また、DeepFaceおよび各モデルのライセンス・利用条件についても、実運用前に確認してください。

## License

ライセンスについては、必要に応じて追加してください。
