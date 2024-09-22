# -*- coding: utf-8 -*-
# 必要なライブラリをインポート
import sys
import cv2
import numpy as np
import numpy as np
import pygetwindow as gw
import glob
import sys
import argparse
import subprocess
from time import sleep
import imageio.v3 as iio #can treat gif with alpha channel https://stackoverflow.com/questions/67454019/how-to-read-gif-with-alpha-channel-in-python-opencv

import openvino as ov
core = ov.Core()

#import onnx
#import onnxruntime as ort

from windows_capture import WindowsCapture, Frame, InternalCaptureControl
from PySide6.QtWidgets import QApplication, QWidget, QLabel
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap
import threading
import ctypes
from ctypes import wintypes
import time

# Check if ffmpeg is installed (for rtmp streaming)
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    ffmpeg_available = True
except:
    ffmpeg_available = False

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

# コマンドライン引数を解析
parser = argparse.ArgumentParser(description='AVATARIAN - Face replacement program.')
group = parser.add_mutually_exclusive_group()
group.add_argument('-window', type=str, help='Name of the window to capture.')
group.add_argument('-camera', type=int, help='Number of the camera to use.')
parser.add_argument('-rtmp', type=str, help='URL of RTMP.')
# NPU, CPU, GPUの引数を追加し、デフォルト値をCPUに設定
parser.add_argument('-device', choices=['CPU', 'GPU','NPU'], default='CPU', help='Device to use for processing (default: CPU).')
parser.add_argument('--detection_model', '-dm', type=str, default='./models/opencv_zoo/face_detection_yunet/face_detection_yunet_2023mar.onnx',
                    help="Usage: Set face detection model type, defaults to './models/opencv_zoo/face_detection_yunet/face_detection_yunet_2023mar.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', '-cth', type=float, default=0.6,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.6. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
parser.add_argument('--recognition_model', '-rm', type=str, default='./models/original/taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml',
                    help="Usage: Set face detection model type, defaults to './models/original/taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml'.")
parser.add_argument('--compare_face_method', '-cfm',  choices=['COSINE', 'L2'], default='L2',
                    help="Usage: Set calcuration method for face comparison, defaults to 'L2'. L2 is suitable for dlib and COSINE is suitable for sface")
parser.add_argument('--compare_face_tolerance', '-cft', type=float, default=0.4,
                    help='Usage: Set tolerance for face comparison, defaults to 0.4 (for L2). When you set compare_face_method to COSINE:[0-1, 1:strict], L2:[0:strict]')
parser.add_argument('--face_alignment', '-fa', choices=['Yes','No'], default='Yes',
                    help="Usage: Set 'Yes' if you use face alignment when compare face, defaults to 'Yes'.")
parser.add_argument('--specific_avatar_image', '-sai', type=str, default="",
                    help="Usage: Set 1 specific avatar image, by default, images in 'avatars' folder are used.")
#parser.add_argument('--save', '-s', action='store_true',
#                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
#parser.add_argument('--vis', '-v', action='store_true',
#                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')

args, unknown = parser.parse_known_args()

# Windows API 定数の定義
HWND_BOTTOM = 1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
GW_HWNDPREV = 3

# Windows API 関数の定義
user32 = ctypes.windll.user32

# GetWindow関数のプロトタイプを定義
GetWindow = user32.GetWindow
GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
GetWindow.restype = wintypes.HWND

def get_previous_window(target_hwnd):
    prev_hwnd = GetWindow(target_hwnd, GW_HWNDPREV)
    return prev_hwnd

def set_window_pos(next_hwnd, prev_hwnd, x, y, cx, cy, flags):
    user32.SetWindowPos(next_hwnd, prev_hwnd, x, y, cx, cy, flags)

def convert_color_space(src):
    if len(src.shape) == 3 and src.shape[2] == 4:
        # non-animation with alpha channel
        return cv2.cvtColor(src, cv2.COLOR_BGR2RGBA)
    elif len(src.shape) == 4:
        # gif/png animation with alpha channel
        for i in range(src.shape[0]):
            src[i] =cv2.cvtColor(src[i], cv2.COLOR_BGR2RGBA)
        return src
    else:
        # nothing needed for image without alpha channel
        return src

def overlay(dst, src):
    '''
    dst is image without alpha channel
    this is alpha_composite of dst and src
    use alpha channel for png and gif by reading image with iio.imread(image_path, plugin="pillow", mode="RGBA"
    Thanks to https://blanktar.jp/blog/2015/02/python-opencv-overlay
    '''
    if src.shape[-1] == 4:
        mask = src[:,:,3]  # alpha channels
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # duplicate for R,G,B
        mask = mask / 255  # normalize alpha channnels from 0-255 to 0.0-1.0
        src = src[:,:,:3]  # image
        dst = (dst *(1 - mask))  # Darken the original image based on transparency
        dst += src * mask  # overlay image with transparency to original image
    else:
        dst = src
    return dst.astype(np.uint8)

def get_face_locations(frame, w, h, face_detector, detection_model):
    face_locations = []
    if detection_model == "yunet":
        '''
        OpenCVのFaceDetectorYNを使用して顔検出を行う
        '''

        # 入力サイズを設定
        face_detector.setInputSize((w, h))

        # 顔検出を実行
        _, faces = face_detector.detect(frame)

        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)
                # マイナスの値が出ることがあり、後続のクロップ処理でエラーが出るため、マイナスをゼロに設定
                box = np.maximum(box, 0)
                #face_locations.append(box.tolist())  # left, top, width, height
                face[0:4] = box
                face_locations.append(face) #SFaceのface_recognizer.alignCropを利用するためには元の形式で渡す必要がある。ほかのモデルでの利用時は[:4]で区切る

    elif any(x in detection_model for x in ["adas-0001", "retail"]):
        '''
        adas-0001, retail-0004, retail-0005が指定されたときは、OpenVINOを使用して顔検出を行う
        '''
        # 入力画像の前処理
        if detection_model == "adas-0001":
            input_image = cv2.resize(frame, (672, 384))
            input_image = input_image.transpose((2, 0, 1))  # HWC to CHW
            input_image = input_image.reshape(1, 3, 384, 672)
        elif detection_model == "retail": 
            input_image = cv2.resize(frame, (300, 300))
            input_image = input_image.transpose((2, 0, 1))  # HWC to CHW
            input_image = input_image.reshape(1, 3, 300, 300)
        
        # 推論実行
        results = face_detector(input_image)[face_detector.output(0)]
        
        """
        # 非同期推論の実行（OpenVino利用）
        infer_request = face_detector.create_infer_request()
        infer_request.start_async(input_image)
        infer_request.wait()  # 非同期呼び出しを待つ

        # 結果の取得
        results = infer_request.get_output_tensor().data

        """
        # 結果の後処理
        for detection in results[0][0]:
            if detection[2] > args.conf_threshold:  # 信頼度がargs.conf_threshold(デフォルト0.9)以上の検出結果のみ使用
                x_min = max(int(detection[3] * w), 0)
                y_min = max(int(detection[4] * h), 0)
                x_max = min(int(detection[5] * w), w)
                y_max = min(int(detection[6] * h), h)
                face_locations.append([x_min, y_min, x_max - x_min, y_max - y_min]) # left, top, width, height
    
    elif detection_model == "haarcascades": 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔を検出
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 検出された顔に矩形を描画
        for (x, y, w, h) in faces:
            face_locations.append([x, y, w, h])
    

    return face_locations
    
def get_face_encodings(frame, face_locations, face_recognizer, recognition_model, onnx_session, compiled_recognition_model):
    """
    与えられた画像から、各顔のエンコーディングを返す。

    :param frame: 顔を含む画像
    :param face_locations: 顔の位置情報のリスト
    :param face_recognizer: cv2.FaceRecognizerSFオブジェクト
    :param onnx_session: dlibのCUDAを利用したバージョン
    :param compiled_recogniton_model: dlibのopenVINOバージョン（CPU,GPU,NPU）
    :return: 顔のエンコーディングのリスト
    """
    encodings = []

    if recognition_model == "dlib":
        # dlibの利用
        for face in face_locations:    
            if len(face) ==4:                
                left, top, width, height = face 
            else:
                left, top, width, height = map(int, face[:4]) # YuNetを使ったときfaceはfloatで入ってくるため、intに変換  
            face_roi = frame[top:top+height, left:left+width]
            #print(face_roi.shape)
            # 入力画像の前処理 with dlib
            preprocessed_image = cv2.resize(face_roi, (150,150))
            preprocessed_image = preprocessed_image / 255.0  # 正規化
            input_data = np.expand_dims(preprocessed_image, axis=0).astype(np.float32)

            if onnx_session is None:
                """
                # 非同期推論の実行（OpenVino利用）
                infer_request = compiled_recognition_model.create_infer_request()
                infer_request.start_async(input_data)
                infer_request.wait()  # 非同期呼び出しを待つ

                # 結果の取得
                face_feature = infer_request.get_output_tensor().data[0]
                """
                # 推論実行
                face_feature = compiled_recognition_model(input_data)[compiled_recognition_model.output(0)][0]
            else:
                # ONNX推論(GPU利用)
                input_name = onnx_session.get_inputs()[0].name
                output_name = onnx_session.get_outputs()[0].name
                face_feature = onnx_session.run([output_name], {input_name: input_data})[0][0] #[0]だと配列の配列（[[]]）になっているので[0][0]

            encodings.append(face_feature)

    elif recognition_model == "sface":
        # sfaceモデルの利用
        for face in face_locations:    
            if len(face) == 4 or args.face_alignment == 'No':
                # YuNetで顔のアラインメントをしない場合、およびadas0001で顔の位置を切り出している場合
                # SFaceモデルの入力サイズに合わせてリサイズ
                if len(face) ==4:                
                    left, top, width, height = face 
                else:
                    left, top, width, height = map(int, face[:4]) # YuNetを使ったときfaceはfloatで入ってくるため、intに変換  
                face_roi = frame[top:top+height, left:left+width]
                face_roi = cv2.resize(face_roi, (112, 112))
                face_feature = face_recognizer.feature(face_roi)
                encodings.append(face_feature[0]) #リストのリスト形式になっているため、[0]で要素を取得
            else:            
                # Yunetで顔を切り出している場合。
                # 顔のアライメント（必要な場合）
                aligned_face = face_recognizer.alignCrop(frame, face)
                # 特徴抽出
                #face_feature = face_recognizer.feature(face_roi)
                face_feature = face_recognizer.feature(aligned_face)
                encodings.append(face_feature[0]) #リストのリスト形式になっているため、[0]で要素を取得

    return encodings

def compare_faces(known_face_encodings, face_to_compare, method='COSINE', tolerance=args.compare_face_tolerance):
    """
    return most similar face from known_face_encodings as True
    """
    distances = []
    if len(known_face_encodings) == 0:
        return distances
    elif method == 'COSINE':
        for known_face_encoding in known_face_encodings:
            #distance = face_recognizer.match(face_to_compare, known_face_encoding, cv2.FaceRecognizerSF_FR_COSINE)
            distance = np.dot(face_to_compare, known_face_encoding) / (np.linalg.norm(face_to_compare) * np.linalg.norm(known_face_encoding))
            distances.append(distance)
        max_distance = max(max(distances), tolerance) 
        return [x >= max_distance for x in distances]

    else:
        # L2正規化→性能悪化
        # known_face_encodings = known_face_encodings / np.linalg.norm(known_face_encodings, axis=1, keepdims=True)
        # face_to_compare = face_to_compare / np.linalg.norm(face_to_compare)
        # ユーグリッド距離（L2距離）を算出
        distances = list(np.linalg.norm(known_face_encodings - face_to_compare, axis=1))
        min_distance = min(min(distances), tolerance) 
        return [x <= min_distance for x in distances]


# ウィンドウ名が指定されていないが、未知の引数が存在する場合、それをウィンドウ名とする
if args.window is None and args.camera is None and unknown:
    args.window = unknown[0]

# モデル実行のバックエンドの指定を取得
backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]

if args.device in core.available_devices:
        # ['CPU', 'GPU', 'NPU']
        device = args.device
else:
    device = "CPU"

# Face recognitionモデルの設定
# モデルのコンパイル方法が指定されている場合、利用する

face_recognizer = None
onnx_session = None
compiled_recognition_model = None
recognition_model = ""
if "face_recognition_sface" in args.recognition_model:
    # SFace顔認識器の作成  
    face_recognizer = cv2.FaceRecognizerSF.create(
        model=args.recognition_model,
        config="",
        backend_id=backend_id,
        target_id=target_id
    )
    recognition_model = "sface"

elif args.recognition_model == "taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml":
    recognition_model = "dlib"
    face_recognition_model = core.read_model(args.recognition_model)
    # 動的形状の設定
    face_recognition_model.reshape([1, 150, 150, 3])  # 例: バッチサイズ1の具体的な形状
    # モデルのコンパイル
    compiled_recognition_model = core.compile_model(model=face_recognition_model, device_name=device)

elif all(x in args.recognition_model for x in ["face_recognition_resnet_model_v1", "onnx"]):
    print("not supported dlib onnx")
    sys.exit(1)
    """
    # dlibのonnx変換モデルはなぜかkernel errorが出るため利用しない
    recognition_model = "dlib"
    # ONNXモデルのロードと初期化
    onnx_path = args.recognition_model
    # ONNX Runtimeの設定
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    #providers = ['CPUExecutionProvider']
    # ONNXランタイムのセッションを作成    
    onnx_session = ort.InferenceSession(onnx_path, providers=providers)
    """
else:
    print("face recognition model is not supported (only sface for opencv and dlib for openvino xml are supported)")
    sys.exit(1)
    


# Face detectionモデルの設定
detection_model = ""
if "face_detection_yunet" in args.detection_model:
    face_detector = cv2.FaceDetectorYN.create(
        model=args.detection_model,
        config="",
        input_size=[320, 320],
        score_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        backend_id=backend_id,
        target_id=target_id)
    detection_model = "yunet"
elif all(x in args.detection_model for x in ["face-detection-adas-0001", "xml"]):
    detection_model = "adas-0001"
    face_detection_model = core.read_model(model=args.detection_model)
    face_detector = core.compile_model(model=face_detection_model, device_name=device)
elif all(x in args.detection_model for x in ["face-detection-retail-000", "xml"]): 
    detection_model = "retail"
    face_detection_model = core.read_model(model=args.detection_model)
    face_detector = core.compile_model(model=face_detection_model, device_name=device)
elif args.detection_model == "haarcascades":
    detection_model = "haarcascades"
    # Haar Cascadeのモデルファイルを読み込む
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
else:
    print("face detection model is not supported (only yunet, haarcascades for opencv and adas-0001, retail-0004, retail-0006 for openvino are supported)")
    sys.exit(1)


# Face Compare
compare_face_method = args.compare_face_method
print(compare_face_method)

compare_face_tolerance = args.compare_face_tolerance


# 'avatars' フォルダ内のすべてのjpg, jpeg, png, apng, gifファイルのパスを取得

avatar_images_paths = []
possible_file_types = ["jpg", "jpeg", "png", "apng", "gif"]
if args.specific_avatar_image == "":
    avatar_paths = glob.glob('avatars/*')
    for file_path in avatar_paths:
        file_path = file_path.lower()
        for ext in possible_file_types:
            if file_path.endswith(ext):
                avatar_images_paths += [file_path]
else:
    file_path = args.specific_avatar_image.lower()
    for ext in possible_file_types:
        if file_path.endswith(ext):
            avatar_images_paths += [file_path]
    if len(avatar_images_paths) == 0:
        print('Given image file is not able to use. Please use file ["jpg", "jpeg", "png", "apng", "gif"]')

# 画像を格納するためのリストを初期化
avatars = []

# アバター画像を読み込み、リストに追加
avatars_rgba = [cv2.imread(image_path) if image_path.endswith("jpg") or image_path.endswith("jpeg")\
                else iio.imread(image_path, plugin="pillow", mode="RGBA") \
                for image_path in avatar_images_paths]
avatars = [convert_color_space(i) for i in avatars_rgba]

# 顔の特徴に対応するアバター画像を保持する配列と辞書
known_face_encodings = []  # This will hold the numpy array of face encodings
face_to_avatar_index = {}





window_title = ""

# 入力ソースを設定
if args.window:
    # ウィンドウキャプチャを使用
    windows = gw.getWindowsWithTitle(args.window)
    if windows:
        # remove cmd.exe as first choice of the window in case you run main.py from cmd.exe
        if args.window in windows[0].title and "main.py" in windows[0].title:
            windows.pop(0)
        window = windows[0]
        window_title = window.title
        window.restore()  # 最小化されている場合はウィンドウを元に戻す
        window.activate()  # ウィンドウをアクティブにする
        

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("camera w,h:",w,h)

else:
    print('Please specify an input source using -window or -camera.')
    sys.exit(1)

# Every Error From on_closed and on_frame_arrived Will End Up Here
capture = WindowsCapture(
    cursor_capture=False,
    draw_border=False,
    monitor_index=None,
    window_name=window_title,
    )

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

# FPS計測
start_time = time.time()
start_time_f = start_time

frame_count = 0
# 無限ループで映像を処理する
class OverlayApp(QWidget):
    face_locations_updated = Signal(list, list)  # シグナルを定義

    def __init__(self, avatars, windows):
        super().__init__()
        self.avatars = avatars
        self.labels = []
        self.frame = None # Captureフレームを保持
        self.frame_width = 0
        self.frame_height = 0
        self.capture_control = None  # CaptureControlオブジェクトを保持
        self.known_face_encodings = []  # 顔の特徴を保持
        self.face_to_avatar_index = {}  # 顔とアバターの対応を保持
        self.windows = windows
        self.initUI()
        self.face_locations_updated.connect(self.overlay_avatars)  # スロットをシグナルに接続
        self.time_start = time.time()
        self.time_prev_start = time.time()
        self.time_func_start = time.time()
        self.time_func_end = time.time()

        # タイマーを設定して定期的にウィンドウの位置とサイズを更新
        self.timer = QTimer(self)
        #self.timer.timeout.connect(self.get_face_information)
        self.timer.timeout.connect(self.check_window_geometry)
        self.timer.start(13) # 80FPSで更新

    def initUI(self):
        self.setWindowTitle("Avatarian")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowTransparentForInput)

    def get_face_information(self):
        # face detection
        self.time_func_start = time.time()
        face_locations = get_face_locations(self.frame, self.frame_width, self.frame_height, face_detector, detection_model) #left, top, width, height
        # print("Got face locations")  # デバッグ情報
        self.time_func_end = time.time()
        print(f'Face detection  :{self.time_func_end - self.time_func_start:.5f} sec, model:{detection_model}')

        # face recognition
        self.time_func_start = time.time()
        if len(self.avatars) > 1:
            face_encodings = get_face_encodings(self.frame, face_locations, face_recognizer, recognition_model, onnx_session, compiled_recognition_model)
        else:
            face_encodings = [1]*len(face_locations)
        self.time_func_end = time.time()
        print(f'Face recognition:{self.time_func_end - self.time_func_start:.5f} sec, model:{recognition_model}')

        # print("Got face encodings")  # デバッグ情報
        #return face_locations, face_encodings
        self.face_locations_updated.emit(face_locations, face_encodings)

    def check_window_visibility(self):
        # ウィンドウがスクリーン上に表示されているか確認
        if self.windows.isMinimized or self.windows.visible !=1:
            # print("invisible",self.windows.isMinimized, self.windows.visible) #デバッグ用      
            if self.isVisible():
                # print("Window is not visible. Hiding overlay.") #デバッグ用
                self.hide()
        else:
            # print("visible",self.windows.isMinimized, self.windows.visible, self.isVisible()) #デバッグ用
            if not self.isVisible():
                # print("Window is visible. Showing overlay.") #デバッグ用
                self.show()
            # ウィンドウハンドルを取得してZオーダーを設定
            try:
                hwnd_overlay = int(self.winId())  # オーバーレイウィジェットのハンドルを取得
                hwnd_target = int(self.windows._hWnd)  # ターゲットウィンドウのハンドルを取得
                
                # target_windowの直前のウィンドウを取得
                prev_hwnd = get_previous_window(hwnd_target)

                if prev_hwnd:
                    # target_windowが最前面でない場合、その直前に配置
                    set_window_pos(hwnd_overlay, prev_hwnd, 0, 0, 0, 0,
                                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)

                else:
                    # target_windowが最前面の場合、最前面に配置
                    set_window_pos(hwnd_target, hwnd_overlay, 0, 0, 0, 0,
                                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
            except:
                print("Window handler error")
                sys.exit(1)

    def check_window_geometry(self):
        # ウィンドウの位置とサイズをチェックし、必要に応じて更新
        try:
            # ウィンドウの位置とサイズ確認
            current_geometry = (self.windows.left, self.windows.top, self.windows.width, self.windows.height)
            if self.geometry() != current_geometry:
                self.setGeometry(window.left, window.top, window.width, window.height)
            self.check_window_visibility() 
        except:
            print("Window handler error")
            sys.exit(1)

    def on_frame_arrived(self, frame: Frame, capture_control: InternalCaptureControl):
        #print("Frame arrived")  # デバッグ情報
        self.frame_width, self.frame_height = frame.width, frame.height
        frame_array = np.array(frame.frame_buffer)
        self.frame = cv2.cvtColor(frame_array, cv2.COLOR_BGRA2BGR)
        self.get_face_information()
        # FPSチェック
        self.time_prev_start = self.time_start
        self.time_start = time.time()
        print('FPS:', 1 / (self.time_start - self.time_prev_start))

    @Slot(list, list)
    def overlay_avatars(self, face_locations, face_encodings):
        # 既存のラベルをクリア
        for label in self.labels:
            label.hide()
        self.labels.clear()

        for index, face_encoding in enumerate(face_encodings):
            # Attempt to match each face encoding to known faces
            #matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=compare_face_tolerance)
            if len(self.avatars) > 1:
                matches = compare_faces(self.known_face_encodings, face_encoding, method=compare_face_method, tolerance=compare_face_tolerance)
                if any(matches):
                    matched_index = matches.index(True)
                    avatar_index = self.face_to_avatar_index[matched_index]
                    avatar_image = self.avatars[avatar_index]
                else:
                    if len(self.avatars) > len(self.known_face_encodings):
                        self.known_face_encodings.append(face_encoding)
                        self.face_to_avatar_index[len(self.known_face_encodings) - 1] = len(self.known_face_encodings) - 1
                        avatar_image = self.avatars[len(self.known_face_encodings) - 1]
                    else:
                        avatar_image = self.avatars[-1]  # Use last one when No more avatars available
            else:
                avatar_image = self.avatars[0]

            # Get the face location and calculate the new size and position
            left, top, width, height = face_locations[index][:4]
            
            # Decide by how much you want to increase the size of the avatar, for example 1.5 times.
            scale_factor = 1.5
            scaled_face_width = int(width * scale_factor)
            scaled_face_height = int(height * scale_factor)

            # Calculate new position to center the avatar over the face location
            top_new = max(top - (scaled_face_height - height) // 2, 0)
            left_new = max(left - (scaled_face_width - width) // 2, 0)
            
            # Resize and overlay the avatar on the frame
            scaled_avatar = avatar_image.scaled(scaled_face_width, scaled_face_height)
            label = QLabel(self)
            label.setPixmap(scaled_avatar)
            label.resize(scaled_face_width, scaled_face_height)
            label.move(left_new, top_new)
            label.show()
            self.labels.append(label)
            """for animation ,https://blog.ssokolow.com/archives/2019/08/14/displaying-an-image-or-animated-gif-in-qt-with-aspect-ratio-preserving-scaling/
            if len(avatar_image.shape) == 3 and avatar_image.shape[2] == 4:
                # image with alpha channel (gif, png)
                #frame[bottom_new:top_new, left_new:right_new] = overlay(frame[bottom_new:top_new, left_new:right_new], cv2.resize(avatar_image, new_dimensions))
                scaled_avatar = avatar_image.scaled(scaled_face_width, scaled_face_height)
                label = QLabel(self)
                label.setPixmap(scaled_avatar)
                label.resize(scaled_face_width, scaled_face_height)
                label.move(left_new, top_new)
                label.show()
                self.labels.append(label)

            elif len(avatar_image.shape) == 4:
                # animation (with alpha channel; gif, apng)
                f = frame_count % avatar_image.shape[0]
                #frame[bottom_new:top_new, left_new:right_new] = overlay(frame[bottom_new:top_new, left_new:right_new], cv2.resize(avatar_image[f], new_dimensions))
                scaled_avatar = avatar_image.scaled(scaled_face_width, scaled_face_height)
                label = QLabel(self)
                label.setPixmap(scaled_avatar)
                label.resize(scaled_face_width, scaled_face_height)
                label.move(left_new, top_new)
                label.show()
                self.labels.append(label)
            else:
                # image (jpg, jpeg)
                #frame[bottom_new:top_new, left_new:right_new] = cv2.resize(avatar_image, new_dimensions)
                scaled_avatar = avatar_image.scaled(scaled_face_width, scaled_face_height)
                label = QLabel(self)
                label.setPixmap(scaled_avatar)
                label.resize(scaled_face_width, scaled_face_height)
                label.move(left_new, top_new)
                label.show()
                self.labels.append(label)
            """
    
    def closeEvent(self, event):
        if self.capture_control:
            self.capture_control.stop()  # キャプチャセッションを停止
        QApplication.quit()  # アプリケーションを終了する

# Windows Captureの設定
@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    # フレームの処理を120FPSに制限
    if not hasattr(on_frame_arrived, "last_time"):
        on_frame_arrived.last_time = time.time()
    current_time = time.time()
    if current_time - on_frame_arrived.last_time >= 1 / 120:
        overlay_window.on_frame_arrived(frame, capture_control)
        on_frame_arrived.last_time = current_time
        
# Session Will End After This Function Ends
@capture.event
def on_closed():
    print("Capture Session Closed")

def start_capture(overlay_window):
    overlay_window.capture_control = capture.start_free_threaded()  # CaptureControlオブジェクトを取得

if __name__ == "__main__":
    app = QApplication(sys.argv)
    avatars = [QPixmap(i) for i in avatar_images_paths]

    overlay_window = OverlayApp(avatars, window)
    overlay_window.show()

    # 新しいスレッドでキャプチャを開始
    capture_thread = threading.Thread(target=start_capture, args=(overlay_window,))
    capture_thread.start()

    sys.exit(app.exec())



# 終了時の処理
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
print(f"Device: {args.device}, Over All FPS: {fps:.2f}")

# 入力ソースと出力先を閉じる
if args.window:
    cv2.destroyAllWindows()
else:
    cap.release()
if process:
    process.terminate()
    process.wait()

