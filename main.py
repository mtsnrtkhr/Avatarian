# 必要なライブラリをインポート
import cv2
import face_recognition
import numpy as np
import pygetwindow as gw
import pyautogui
import glob
import sys

# コマンドライン引数をチェック
if len(sys.argv) < 2:
    print("ウィンドウのタイトルを引数として指定してください。")
    sys.exit(1)

window_title = sys.argv[1]  # コマンドライン引数からウィンドウのタイトルを取得

# 'avatars' フォルダ内のすべての '.jpg' および '.png' ファイルのパスを取得
avatar_images_paths = glob.glob('avatars/*.jpg') + glob.glob('avatars/*.png')

# 画像を格納するためのリストを初期化
avatars = []

# アバター画像を読み込み、リストに追加
avatars = [cv2.imread(image_path) for image_path in avatar_images_paths]

# 顔の特徴に対応するアバター画像を保持する配列と辞書
known_face_encodings = []  # This will hold the numpy array of face encodings
face_to_avatar_index = {}

# タイトルが指定されたウィンドウを取得
windows = gw.getWindowsWithTitle(window_title)
if windows:
    window = windows[0]
    window.restore()  # 最小化されている場合はウィンドウを元に戻す
    window.activate()  # ウィンドウをアクティブにする
else:
    print(f"'{window_title}' タイトルのウィンドウが見つかりませんでした。")
    sys.exit(1)  # ウィンドウが見つからなかった場合にプログラムを終了

# 無限ループで映像を処理する
while True:
    # ウインドウのbboxを取得し、画面をキャプチャする
    x, y, w, h = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
                continue  # No more avatars available


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

    '''
    # 顔の特徴抽出をしない場合
    # 検出した顔の数だけ繰り返す
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # avatars リストの長さで割った余りをインデックスとして使用
        index = i % len(avatars)
        # 選択したアバター画像をリサイズする
        avatar_resized = cv2.resize(avatars[index], (right - left, bottom - top))
        # 顔の位置にアバター画像を重ねる
        frame[top:bottom, left:right] = avatar_resized

    '''

    # 画像を表示する
    cv2.imshow("Avatarian", frame)
    # qキーを押すと終了する
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# カメラとウィンドウを閉じる
cv2.destroyAllWindows()

