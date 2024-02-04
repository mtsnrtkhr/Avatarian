[日本語版](README_JP.md)  
# Avatarian
Avatarian is an app that transforms the faces of people in web conferences into avatars in real-time.

## Demo image
![Demo Image](media/avatarian_demo.jpg "Avatarian on Around")
This is a demo image of Avatarian on "Around" web meeting App.

## How to Use
This application offers a variety of options to customize your experience:

1. **Application Selection**: You can specify the application you want to use with this app.

2. **Camera Selection**: The app allows you to choose the camera you want to use.

3. **Streaming Output**: There's an option to stream your output, providing a live feed of your usage.

Here's a step-by-step guide on how to use these options:

### 1. **Application Selection**
To specify the application, run the command
```shell
python main.py WINDOW_NAME
```
or
```shell
python main.py -window WINDOW_NAME
```
Please replace WINDOW_NAME with the name of the window you want to capture.  
Note that the application can only capture the window that is in the foreground.”

### 2. **Camera Selection**
To choose the camera, run the command
```shell
python main.py -camera CAMERA_NUMBER
```
Here, CAMERA_NUMBER is the number assigned to the camera you want to use.  
For example, if you want to use the first camera, you would enter python main.py -camera 0.

### 3. **Streaming Output**
To stream your output for RTMP, run the command 
```shell
python main.py (-window WINDOW_NAME / -camera CAMERA_NUMBER) -rtmp YOUR_RTMP_STREAMING_URL
```
Replace WINDOW_NAME with the name of the window you want to capture or  
CAMERA_NUMBER with the number of the camera you want to use,  
and replace YOUR_RTMP_STREAMING_URL with your actual RTMP streaming URL.

## Avatars
Currently, this app supports avatar creation using image files only, not 3D models.  
Please save your preferred avatar image in the 'avatars' folder.  
The image should be in either jpg or png format.  
Avatars are assigned in the order that faces are recognized,  
corresponding to the alphabetical order of their names.  
If there aren't enough avatars, the last image will be assigned to the remaining faces.


## How to Install
You can install this app on Windows OS in 5 steps from scratch:  
1. Install a C++ compiler.  
    I recommend the "Microsoft C++ Build Tool".  
2. Install Cmake.  
    This step is optional if you're using the "Microsoft C++ Build Tool".  
    If you do install it, please ensure it's added to your PATH in the environmental variables.
3. Install Python (version 3.10 or higher).  
    Make sure to add it to your PATH in the environmental variables.
4. Download "Avatarian" from this repository as a zip file and extract it.
5. Install libraries of Python by the script below in cmd.exe.  
```shell
pip install opencv-python face_recognition pyautogui pygetwindow setuptools imageio
```

### About C++ Compiler
Many of you may use this app for work, especially for online meetings.  
However, please be mindful of the C++ compiler license when doing so.  
While Microsoft Visual Studio Community is suitable for personal use,  
I recommend the [Microsoft C++ Build Tool](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for commercial use.  
For more information on the license, please refer to the following article:
[Updates to Visual Studio Build Tools license for C and C++ Open-Source projects](https://devblogs.microsoft.com/cppblog/updates-to-visual-studio-build-tools-license-for-c-and-cpp-open-source-projects/)(Published on August 18th, 2022)  

Please note that by default, installing the Microsoft C++ Build Tool will also install the C++ compiler and Cmake.

