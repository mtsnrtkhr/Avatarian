# -*- coding: utf-8 -*-
# 必要なライブラリをインポート
import cv2
import face_recognition
import numpy as np
import pygetwindow as gw
import pyautogui
import glob
import sys
import argparse
import subprocess
from time import sleep

# Check if ffmpeg is installed (for rtmp streaming)
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    ffmpeg_available = True
except:
    ffmpeg_available = False

# コマンドライン引数を解析
parser = argparse.ArgumentParser(description='AVATARIAN - Face replacement program.')
group = parser.add_mutually_exclusive_group()
group.add_argument('-window', type=str, help='Name of the window to capture.')
group.add_argument('-camera', type=int, help='Number of the camera to use.')
parser.add_argument('-rtmp', type=str, help='URL of RTMP.')
args, unknown = parser.parse_known_args()

# ウィンドウ名が指定されていないが、未知の引数が存在する場合、それをウィンドウ名とする
if args.window is None and args.camera is None and unknown:
    args.window = unknown[0]

# 'avatars' フォルダ内のすべての '.jpg' および '.png' ファイルのパスを取得
avatar_images_paths = glob.glob('avatars/*.jpg') + glob.glob('avatars/*.png')

# 画像を格納するためのリストを初期化
avatars = []

# アバター画像を読み込み、リストに追加
avatars = [cv2.imread(image_path) for image_path in avatar_images_paths]

# 顔の特徴に対応するアバター画像を保持する配列と辞書
known_face_encodings = []  # This will hold the numpy array of face encodings
face_to_avatar_index = {}

# 入力ソースを設定
if args.window:
    # ウィンドウキャプチャを使用
    windows = gw.getWindowsWithTitle(args.window)
    if windows:
        # remove cmd.exe as first choice of the window in case you run main.py from cmd.exe
        if args.window in windows[0].title and "main.py" in windows[0].title:
            windows.pop(0)
        window = windows[0]
        window.restore()  # 最小化されている場合はウィンドウを元に戻す
        window.activate()  # ウィンドウをアクティブにする
        w, h = int(window.width), int(window.height)
    else:
        print(f"'{args.window}' 指定した名前のウィンドウが見つかりませんでした。")
        print("Active titles are below. Please select the word(s) in it when you fail to start.\n")
        for wnd in gw.getAllWindows():
            if len(wnd.title):
                print(wnd.title) 
        sys.exit(1)  # ウィンドウが見つからなかった場合にプログラムを終了

elif args.camera is not None:
    # カメラを使用
    cap = cv2.VideoCapture(args.camera)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
else:
    print('Please specify an input source using -window or -camera.')
    sys.exit(1)

# 出力先を設定
if args.rtmp and ffmpeg_available:
    # RTMPを使用
    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '{}x{}'.format(w, h), '-r', '25', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-f', 'flv', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2'.format(w), args.rtmp]
    # ffmpegプロセスの開始
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    stream = True
elif args.rtmp:
    print('Can not start rtmp streaming. Please install ffmpeg.')
    if args.window:
        window.close()
    else:
        cap.release()
    sys.exit(1)
else:
    # 画面に表示
    stream = None

# 無限ループで映像を処理する
while True:
    # 入力ソースからフレームを取得
    if args.window:
        x, y, w, h = window.left, window.top, window.width, window.height
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cap.read()
        if not ret:
            break
    
    # 画像から顔の位置を検出する
    face_locations = face_recognition.face_locations(frame)

    # 画像から顔の特徴を検出する
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop over each face found in the frame
    for index, face_encoding in enumerate(face_encodings):
        # Attempt to match each face encoding to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        
        if any(matches):
            matched_index = matches.index(True)
            avatar_index = face_to_avatar_index[matched_index]
            avatar_image = avatars[avatar_index]
        else:
            if len(avatars) > len(known_face_encodings):
                known_face_encodings.append(face_encoding)
                face_to_avatar_index[len(known_face_encodings) - 1] = len(known_face_encodings) - 1
                avatar_image = avatars[len(known_face_encodings) - 1]
            else:
                avatar_image = avatars[-1]  # Use last one when No more avatars available

        # Get the face location and calculate the new size and position
        top, right, bottom, left = face_locations[index]
        face_width = right - left
        face_height = bottom - top
        # Decide by how much you want to increase the size of the avatar, for example 1.5 times.
        scale_factor = 1.5
        new_width = int(face_width * scale_factor)
        new_height = int(face_height * scale_factor)

        # Calculate new position to center the avatar over the face location
        top_new = max(top - (new_height - face_height) // 2, 0)
        right_new = min(right + (new_width - face_width) // 2, frame.shape[1])
        bottom_new = min(bottom + (new_height - face_height) // 2, frame.shape[0])
        left_new = max(left - (new_width - face_width) // 2, 0)

        # Resize and overlay the avatar on the frame
        new_dimensions = (right_new - left_new, bottom_new - top_new)
        avatar_resized = cv2.resize(avatar_image, new_dimensions)
        frame[top_new:bottom_new, left_new:right_new] = avatar_resized
    
    # 出力先にフレームを送信
    if stream:
        cv2.imshow("Avatarian", frame)
        process.stdin.write(frame.tobytes())
        sleep(0.01)
    else:
        cv2.imshow("Avatarian", frame)

    # qキーを押すと終了する
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 入力ソースと出力先を閉じる
if args.window:
    window.close()
else:
    cap.release()
if process:
    process.terminate()
    process.wait()

