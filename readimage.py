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
        ##cv2.imshow("frame", frame)
        ##cv2.waitKey(50)
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

def convert_color_space(img):
    if len(img.shape) == 3 and img.shape[2] == 4:
        # non-animation with alpha channel
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif len(img.shape) == 4:
        # gif/png animation with alpha channel
        for i in range(img.shape[0]):
            img[i] =cv2.cvtColor(img[i], cv2.COLOR_BGR2RGBA) 
        return img
    else:
        # image without alpha channel
        return img
    



def overlay(dest, src):
    '''
    use alpha channel for png and gif by reading image with iio.imread(image_path, plugin="pillow", mode="RGBA"
    Thanks to https://blanktar.jp/blog/2015/02/python-opencv-overlay
    '''
    # print(dest.shape, src.shape)
    if src.shape[-1] == 4:
        mask = src[:,:,3]  # alpha channels
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # duplicate for R,G,B
        mask = mask / 255  # normalize alpha channnels from 0-255 to 0.0-1.0
        src = src[:,:,:3]  # image
        # cv2.imshow("src", src)
        # cv2.waitKey(1000)

        dest = (dest *(1 - mask))  # 透過率に応じて元の画像を暗くする。
        # cv2.imshow("dest", dest)
        # cv2.waitKey(1000)

        dest += src * mask  # 貼り付ける方の画像に透過率をかけて加算。
    else:
        dest = src
    return dest



# 'avatars' フォルダ内のすべてのjpg, jpeg, png, gifファイルのパスを取得
avatar_images_paths = []
avatar_paths = glob.glob('avatars/gitignore_avatars/*')
possible_file_types = ["jpg", "jpeg", "png", "gif"]
for file_path in avatar_paths:
    # print(file_path)
    for ext in possible_file_types:
        if file_path.lower().endswith(ext):
            avatar_images_paths += [file_path]
# for file_path in avatar_images_paths:
#    print(file_path)

# 画像を格納するためのリストを初期化
avatars = []

# アバター画像を読み込み、リストに追加
avatars = [iio.imread(image_path) for image_path in avatar_images_paths]
#avatars_rgba = [iio.imread(image_path, plugin="pillow", mode="RGBA") for image_path in avatar_images_paths]
avatars_rgba = [cv2.imread(image_path) if image_path.endswith("jpg") else iio.imread(image_path, plugin="pillow", mode="RGBA") for image_path in avatar_images_paths]


avatars_rgba_cvtcolors = [convert_color_space(i) for i in avatars_rgba]

'''
for i in avatars_rgba_cvtcolors:
    print(i.shape)
    if len(i.shape) == 4:
        for j in range(i.shape[0]):
            cv2.imshow("anime", i[j])
            cv2.waitKey(200)
    else:
        cv2.imshow("img",i)
        cv2.waitKey(1000)
'''

background = iio.imread("./avatars/gitignore_avatars/grant-ritchie-x1w_Q78xNEY-unsplash.jpg")
background = cv2.resize(background,(600,400))
#cv2.imshow("bg", background)
#cv2.waitKey(500)
#print(len(avatars))

cnt = 0
x_pos = [100,200]
y_pos = [200,300] 
x_width = x_pos[1] - x_pos[0]
y_height = y_pos[1] - y_pos[0]
while True:
    for i in avatars_rgba_cvtcolors:
        print(i.shape)
        bg = background.copy()
        if len(i.shape) == 3 and i.shape[2] == 4:
            bg[x_pos[0]:x_pos[1],y_pos[0]:y_pos[1]] = overlay(bg[x_pos[0]:x_pos[1],y_pos[0]:y_pos[1]], cv2.resize(i,(x_width, y_height)))

        elif len(i.shape) == 4:
            frame = cnt % i.shape[0]
            bg[x_pos[0]:x_pos[1],y_pos[0]:y_pos[1]] = overlay(bg[x_pos[0]:x_pos[1],y_pos[0]:y_pos[1]], cv2.resize(i[frame], (x_width, y_height)))
        else:
            bg[x_pos[0]:x_pos[1],y_pos[0]:y_pos[1]] = cv2.resize(i,(x_width, y_height))
        cv2.imshow("bg",bg)
        cv2.waitKey(1000)
    cnt += 1


'''
for img in avatars:
    print(img.shape)
    if len(img.shape) == 4:
        # gif/png animation

        #print(img[0].shape) ndarray with (num_frames, height, width, channel=3)
        #cv2.imshow("img", img[0])
        #cv2.waitKey(1000)
        img_rgb =cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB) 
        #print(img_rgb.shape)
        #cv2.imshow("img2", img_rgb)
        #cv2.waitKey(1000)
        img = cv2.resize(img_rgb, (100,100))

        bg = background.copy()
        bg[0:100,0:100] = img
        cv2.imshow("bg",bg)
        cv2.waitKey(1000)

    elif img.shape[2] == 4:
        # png ndarray with (height, width, channel=4)
        #cv2.imshow("img", img)
        #cv2.waitKey(1000)
        img_rgb =cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        #cv2.imshow("img", img_rgb)
        #print("png" , img_rgb.shape)
        #cv2.waitKey(1000)
        img = cv2.resize(img_rgb, (100,100))

        bg = background.copy()
        bg[0:100,0:100] = img
        cv2.imshow("bg",bg)
        cv2.waitKey(1000)

    else:
        # jpg, jpge
        img_rgb =cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        #cv2.imshow("img", img_rgb)
        #cv2.waitKey(1000)

        img = cv2.resize(img_rgb, (100,100))

        bg = background.copy()
        bg[0:100,0:100] = img
        cv2.imshow("bg",bg)
        cv2.waitKey(1000)
'''

'''
for img in avatars_rgba:
    print(img.shape)
    if len(img.shape) == 3:
        #non-animation
        if img.shape[2]== 3: 
            # no-alpha channel = jpg/jpeg witch read cv2, so don't need to cv2.cvtColor

            #cv2.imshow("img", img)
            #cv2.waitKey(1000)
            
            img = cv2.resize(img, (100,100))

            bg = background.copy()
            bg[300:400,200:300] = img
            cv2.imshow("bg",bg)
            cv2.waitKey(1000)

        elif img.shape[2] == 4:
            # non-animation, with-alpha channel png/gif (height, width, channel=4)
            #print(img[:,:,3]) #alpha channels

            #cv2.imshow("img", img)
            #cv2.waitKey(1000)
            img_rgb =cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) # iio need color conversion
            #cv2.imshow("img", img_rgb)
            #print("png" , img_rgb.shape)
            #cv2.waitKey(1000)
            img_resized = cv2.resize(img_rgb, (100,100))

            bg = background.copy()
            bg[300:400,200:300] = overlay(bg[300:400,200:300], img_resized)
            cv2.imshow("bg",bg)
            cv2.waitKey(1000)

    elif len(img.shape) == 4:
        # gif/png animation with alpha channel
        #print(img[0][:,:,3])
    
        #print(img[0].shape) ndarray with (num_frames, height, width, channel=3)
        #cv2.imshow("img", img[0])
        #cv2.waitKey(1000)
        for f in range(img.shape[0]):
            print(f)
            img_rgb =cv2.cvtColor(img[f], cv2.COLOR_BGR2RGBA) 
            #print(img_rgb.shape)
            #cv2.imshow("img2", img_rgb)
            #cv2.waitKey(1000)
            img_resized = cv2.resize(img_rgb, (100,100))

            bg = background.copy()
            bg[300:400,200:300] = overlay(bg[300:400,200:300], img_resized)
            cv2.imshow("bg",bg)
            cv2.waitKey(10)

    else:
        print("False")


'''
