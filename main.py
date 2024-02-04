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
import imageio.v3 as iio #can treat gif with alpha channel https://stackoverflow.com/questions/67454019/how-to-read-gif-with-alpha-channel-in-python-opencv

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

def read_gif_animation(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        #print(len(frame))
        #np.savetxt("frame.csv", frame.reshape((3,-1)), fmt="%s",header=str(frame.shape))
        #cv2.imshow("frame", frame)
        #cv2.waitKey(50)
    cap.release()
    return frames

def is_gif_animation(avatar_image):
    try:
        frames = len(avatar_image)
        # check if image is list(gif;animationframes) or numpy.ndarray(jpeg/png;image)
        if isinstance(avatar_image, list):
            return True, frames
        else:
            return False, 0
    except:
        return False, 0

def image_in_gif_frame(avatar_image, frame_count):
    g = False
    f = 0
    g, f = is_gif_animation(avatar_image)
    if g:
        f = frame_count % f
        avatar_image = avatar_image[f]
    return avatar_image
    
def overlay_image(background, overlay):
    # Create a mask for non-white pixels in the overlay image
    mask = cv2.inRange(overlay, (255,255,255), (255,255,255))
    # Overlay the non-white parts of the overlay image onto the background
    result = cv2.bitwise_and(background, background, mask=mask)
    result += cv2.bitwise_and(overlay, overlay, mask=~mask)
    return result

# ウィンドウ名が指定されていないが、未知の引数が存在する場合、それをウィンドウ名とする
if args.window is None and args.camera is None and unknown:
    args.window = unknown[0]

# 'avatars' フォルダ内のすべての '.jpg' および '.png' ファイルのパスを取得
avatar_images_paths = glob.glob('avatars/*.jpg') + glob.glob('avatars/*.png') 
avatar_gif_images_paths = glob.glob('avatars/*.GIF')

# 画像を格納するためのリストを初期化
avatars = []

# アバター画像を読み込み、リストに追加
avatars =  [read_gif_animation(image_path) for image_path in avatar_gif_images_paths]
avatars += [cv2.imread(image_path) for image_path in avatar_images_paths]
#avatars += [iio.imread(image_path) for image_path in avatar_images_paths]

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

frame_count = 0
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
#            cv2.imshow("1",avatar_image)
#            cv2.waitKey(1000)
            avatar_image = image_in_gif_frame(avatar_image, frame_count)
        else:
            if len(avatars) > len(known_face_encodings):
                known_face_encodings.append(face_encoding)
                face_to_avatar_index[len(known_face_encodings) - 1] = len(known_face_encodings) - 1
                avatar_image = avatars[len(known_face_encodings) - 1]
#                cv2.imshow("2",avatar_image)
#                cv2.waitKey(1000)
                avatar_image = image_in_gif_frame(avatar_image, frame_count)

            else:
                avatar_image = avatars[-1]  # Use last one when No more avatars available
                avatar_image = image_in_gif_frame(avatar_image, frame_count)
#                cv2.imshow("3",avatar_image)
#                cv2.waitKey(1000)

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
        avatar_overlay = overlay_image(frame[top_new:bottom_new, left_new:right_new], avatar_resized)
        #frame[top_new:bottom_new, left_new:right_new] = avatar_resized
        frame[top_new:bottom_new, left_new:right_new] = avatar_overlay
    
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

    frame_count += 1

# 入力ソースと出力先を閉じる
if args.window:
    window.close()
else:
    cap.release()
if process:
    process.release()

