# Avatarian
Avatarian（アバタリアン）は、ウェブ会議での人々の顔をリアルタイムでアバターに変換するアプリです。

## デモ画像
![Demo Image](media/avatarian_demo.jpg "Avatarian on Around")  
これは、ウェブ会議アプリ「Around」でのAvatarianのデモ画像です。

## 使い方
このアプリケーションでは、以下のオプションを利用してカスタマイズが可能です。

### 1. アプリケーション選択
置き換えたい顔が映っているアプリケーションを１つ指定できます。  
以下のコマンドを実行します：
```shell
python main.py WINDOW_NAME
```
または
```shell
python main.py -window WINDOW_NAME
```
```WINDOW_NAME```をキャプチャしたいウィンドウの名前に置き換えてください。

### 2. 顔検出モデル
下記の顔の検出のモデルを指定できます。
- YuNet:比較的高速で、小さい顔の検出にも強い（-cthを0.6程度まで下げるとほぼ検出できる。ただし、誤検出も増えてイラストも顔と認識する）
- face-detection-adas:高速で、斜め向きの顔の検出にも強い。小さい顔は検出できない
- face-detection-retail:かなり高速だが、検出精度が若干低い。小さい顔は検出できない。
- Haar Cascade: OpenCVに同梱されている機械学習のモデル。上記に比べたメリットは様々な環境で動くこと。  

各モデルのライセンスは*ライセンス*の項目を参照ください。

#### 顔検出モデルのオプション
```--detection_model```, ```-dm``` : 上記のモデルのパス（YuNetはonnxファイル, face-detection-adas, face-detection-retailはxmlファイル）、Harr Cascadeは'haarcascades'を指定してください。デフォルト値：```'./models/opencv_zoo/face_detection_yunet/face_detection_yunet_2023mar.onnx'```  
```--conf_threshold```, ```-cth``` (0~1): 数値が高いほど顔と認識できるものに限って検出します。ぼやけた顔も検出するには数値を下げる必要があります。デフォルト値：```0.6```  
```--nms_threshold``` (0~1)：NMSという手法で、重複する検出ボックスの削除に利用するパラメーターです。指定した値より信頼度が高いもののみ残します。デフォルト値：```0.3```（YuNetのみ）  
```--top_k```：NMSの処理前に上位何個の検出ボックスを使うかを指定するパラメーターです。少ないほど高速になります。デフォルト値：```5000```（YuNetのみ）  

### 3. 顔認識モデル
顔の認識のモデルを指定できます
- dlib:有名な顔認識のモデルに日本人の顔を追加学習させたもの。これを独自にOpenVino用のフォーマットにして高速化したもの。
- SFace:認識精度が高いモデルと言われているが、日本人だとあまり変わらないか、劣る可能性がある。速度はdlibのOpenVino版の倍以上かかる。

各モデルのライセンスは*ライセンス*の項目を参照ください。

#### 顔認識モデルのオプション
```--recognition_model```, ```-rm```：上記のモデルのパス（SFaceはonnxファイル, dlibはxmlファイル）を指定してください。デフォルト値：```'./models/original/taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml'```  
```--compare_face_method```, ```-cfm```：['L2', 'COSINE'] L2は特徴量の距離、COSINEはベクトルのコサイン類似度で計算。dlibを使うときは"L2"、SFaceは"COSINE"を推奨。デフォルト値：```'L2'```  
```--compare_face_tolerance```, ```-cft```：顔が同じかどうかを判断する指標。L2のときは0が完全一致で、0.4~0.6を推奨、COSINEのときは1が完全一致で、0.7-0.8を推奨。デフォルト値：```0.4```  
```--face_alignment```, ```-fa```：['Yes', 'No'] YuNetを使って検出した顔を縦向きにそろえて切り出す処理。この処理によって顔の認識率が上がる。速度低下はほぼありません。デフォルト値：```'Yes'``` (YuNet-Sfaceの組み合わせのみ)： 　


### 4. アバター
現在、このアプリは画像ファイルを使用したアバター作成のみをサポートしており、3Dモデルはサポートしていません。  
お好みのアバター画像を’avatars’フォルダに保存してください。  
画像はjpg(jpeg)、png(apng)またはgif形式である必要があります（拡張子も対応したものにしてください）。  
アバターは、顔が認識された順序に対応して、名前のアルファベット順に割り当てられます。  
アバターが足りない場合、最後の画像が残りの顔に割り当てられます。  

#### アバターのオプション
```--specific-avatar-image```, ```-sai``` : "画像のパス"を指定し、すべての顔を指定の画像に置き換えます。  
この場合、顔認識モデルを利用しないため、若干高速になります。

#### アバターのライセンス
アバターの画像は生成AIによって作成されたものです。CC0 1.0 Universalライセンスのもとで提供されます。

## インストール方法
このアプリは、Windows OSに対して完全に新規で5ステップでインストールできます：  

1. Python（バージョン3.12）をインストールします。  

2. このリポジトリから"Avatarian"をzipファイルとしてダウンロードし、解凍します。

3. pip install -r poetry-requirements.txt で必要なファイルをインストールします

## ライセンス
modelsフォルダに保管されている各モデルはそれぞれライセンスが定められています。

### 外部モデル一覧

| モデル名 | 形式 | サイズ | ライセンス | 出典 | 用途 |
|---|---|---|---|---|---|
| YuNet  (通常/INT8) | ONNX | 98KB/228KB | MIT | OpenCV_zoo/Copyright (c) 2020 Shiqi Yu  | 顔検出 |
| face-detection-adas-0001  (PF16-INT8/PF16/FP32) | bin/xml | 1.6MB/2.3MB/4.2MB | Apache License 2.0 | Open Model Zoo/Copyright (c) 2023-2024 Intel Corporation | 顔検出 |
| face-detection-retail-0004  (PF16-INT8/PF16/FP32) | bin/xml | 0.8MB/1.3MB/2.3MB | Apache License 2.0 | Open Model Zoo/Copyright (c) 2023-2024 Intel Corporation | 顔検出 |
| face-detection-retail-0005  (PF16-INT8/PF16/FP32) | bin/xml | 1.5MB/2.2MB/4.0MB | Apache License 2.0 | Open Model Zoo/Copyright (c) 2023-2024 Intel Corporation | 顔検出 |
| SFace  (通常/INT8) | ONNX | 37MB/9.4MB | Apache License 2.0 | Open Model Zoo | 顔認識 |

**各ライセンスの詳細については、各モデルのディレクトリ内のLICENSEファイルを参照してください。**  

### オリジナルモデル
#### モデル名
- **taguchi_face_recognition_resnet_v1_openvino_fp16_optimized**  
#### 説明
- このモデルは、[dlib-models](https://github.com/davisking/dlib-models/blob/master/README.md)のレポジトリでCC0 1.0 Universalライセンスで公開されている、taguchi_face_recognition_resnet_model_v1.datをベースに、OpenVINO形式に変換し、FP16に量子化することで高速化を図ったものです。  
Taguchi氏によって開発されたtaguchi_face_recognition_resnet_model_v1はオリジナルのdlib_face_recognition_resnet_model_v1を改良し、アジア人顔の認識精度を高めたことが特徴です。

#### 形式
- OpenVINO (bin/xml)

#### サイズ
- 10.9MB

#### ライセンス
- このモデルは、CC0 1.0 Universalライセンスのもとで提供されます。

