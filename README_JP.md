# Avatarian
Avatarian（アバタリアン）は、ウェブ会議での人々の顔をリアルタイムでアバターに変換するアプリです。

## デモ画像
![Demo Image](media/avatarian_demo.jpg "Avatarian on Around")  
これは、ウェブ会議アプリ「Around」でのAvatarianのデモ画像です。

## 使い方
このアプリケーションでは、以下のオプションを利用してカスタマイズが可能です。

1. **アプリケーション選択**：ウェブ会議等のアプリケーションを指定できます。

2. **カメラ選択**：使用するカメラを選択できます。

3. **ストリーミング出力**：Avatarianの画面をRTMPでストリーム出力するオプションがあります。

以下に、これらのオプションの使い方を説明します：

### 1. アプリケーション選択
アプリケーションを指定するには、以下のコマンドを実行します：
```shell
python main.py WINDOW_NAME
```
または
```shell
python main.py -window WINDOW_NAME
```
```WINDOW_NAME```をキャプチャしたいウィンドウの名前に置き換えてください。
このアプリケーションは、前面に表示されているウィンドウのみをキャプチャできます。

### 2. カメラ選択
カメラを選択するには、以下のコマンドを実行します：
```shell
python main.py -camera CAMERA_NUMBER
```
ここで、```CAMERA_NUMBER```は使用したいカメラに割り当てられた番号です。  
例えば、最初のカメラを使用したい場合は、```python main.py -camera 0```と入力します。  

### 3. ストリーミング出力
RTMP用の出力をストリームするには、以下のコマンドを実行します：  
```shell
python main.py (-window WINDOW_NAME / -camera CAMERA_NUMBER) -rtmp YOUR_RTMP_STREAMING_URL
```
アプリケーションかカメラを設定した後に ```-rtmp YOUR_RTMP_STREAMING_URL```を入力します。
```YOUR_RTMP_STREAMING_URL```は実際のRTMPストリーミングURLに置き換えてください。  

## アバター
現在、このアプリは画像ファイルを使用したアバター作成のみをサポートしており、3Dモデルはサポートしていません。  
お好みのアバター画像を’avatars’フォルダに保存してください。  
画像はjpg(jpeg)、png(apng)またはgif形式である必要があります。  
アバターは、顔が認識された順序に対応して、名前のアルファベット順に割り当てられます。  
アバターが足りない場合、最後の画像が残りの顔に割り当てられます。  

## インストール方法
このアプリは、Windows OSに対して完全に新規で5ステップでインストールできます：  

1. C++コンパイラをインストールします。  
"Microsoft C++ Build Tool"を推奨します。

3. Cmakeをインストールします。  
"Microsoft C++ Build Tool"を使用している場合、このステップはオプションです。  
インストールする場合は、環境変数のPATHに追加されていることを確認してください。

4. Python（バージョン3.10以上）をインストールします。  
環境変数のPATHに追加されていることを確認してください。

5. このリポジトリから"Avatarian"をzipファイルとしてダウンロードし、解凍します。

6. 以下のスクリプトでPythonのライブラリをcmd.exeでインストールします。
```shell
pip install opencv-python face_recognition pyautogui pygetwindow setuptools imageio
```

### C++コンパイラについて
多くの方が、特にオンライン会議でこのアプリを業務で使用するかもしれません。  
しかし、その際にはC++コンパイラのライセンスにご注意ください。  
Microsoft Visual Studio Communityは個人使用に適していますが、  
商用利用の場合は [Microsoft C++ Build Tool](https://visualstudio.microsoft.com/visual-cpp-build-tools/) を推奨します。  
ライセンスについての詳細は、以下の記事をご覧ください： [Updates to Visual Studio Build Tools license for C and C++ Open-Source projects](https://devblogs.microsoft.com/cppblog/updates-to-visual-studio-build-tools-license-for-c-and-cpp-open-source-projects/)  
(2022年8月18日公開)  

なお、Microsoft C++ Build Toolをデフォルトでインストールすると、C++コンパイラとCmakeも同時にインストールされます。
