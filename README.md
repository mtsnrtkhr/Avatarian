# Avatarian
Avatarian is an app that transforms the faces of people in web conferences into avatars in real-time.

### Demo image
![Demo Image](media/avatarian_demo.jpg "Avatarian on Around")
This is a demo image of Avatarian on "Around" web meeting App.

### How to Use
To use this application for an app that is currently running, run the command
```shell
python main.py WINDOW_NAME
```
or
```shell
python main.py -window WINDOW_NAME
```
Please replace WINDOW_NAME with the name of the window you want to capture.  
Note that the application can only capture the window that is in the foreground.”

#### Input option
To use this application with your PC’s camera, run the command
```shell
python main.py -camera CAMERA_NUMBER
```
Here, CAMERA_NUMBER is the number assigned to the camera you want to use.  
For example, if you want to use the first camera, you would enter python main.py -camera 0.

#### Output option
If you want to use this application with an RTMP stream as the output, run the command 
```shell
python main.py (-window WINDOW_NAME / -camera CAMERA_NUMBER) -rtmp YOUR_RTMP_STREAMING_URL
```
Replace WINDOW_NAME with the name of the window you want to capture or  
CAMERA_NUMBER with the number of the camera you want to use,  
and replace YOUR_RTMP_STREAMING_URL with your actual RTMP streaming URL.

### Avatars
At this moment you can use only images.  
Please save your favorite avatar image in the avatars folder in either jpg or png format.

### How to install
You can install this app in 5 steps for Windows from completely new.  
1. Install C++ compiler. ("Microsoft C++ Build Tool" is recommended)  
2. Install Cmake. (Optional when you use "Microsoft C++ Build Tool")  
    Please make sure to add PATH to the environmental variables.
3. Install Python (>3.10).   
    Please make sure to add PATH to the environmental variables.
4. Download "Avatarian" from this repository as zip file and extract it.  
5. Install libraries of Python by the script below in cmd.exe.  
```shell
pip install opencv-python face_recognition pyautogui pygetwindow setuptools
```


You may use this app for work.  
Then you need to care about licence of C++ compiler.  
Microsoft Visual Studio Community is OK for personal user but  
I recommend [Microsoft C++ Build Tool](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for commercial use.  
See the article below for the licence.  
[Updates to Visual Studio Build Tools license for C and C++ Open-Source projects](https://devblogs.microsoft.com/cppblog/updates-to-visual-studio-build-tools-license-for-c-and-cpp-open-source-projects/)(August 18th, 2022)  

C++ compiler and Cmake can be installed with Microsoft C++ Build Tool by default.

