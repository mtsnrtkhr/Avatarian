[日本語版](README_JP.md)  
# Avatarian
Avatarian is an app that transforms the faces of people in web conferences into avatars in real-time.

## Demo image
![Demo Image](media/avatarian_demo.jpg "Avatarian on Around")
This is a demo image of Avatarian on "Around" web meeting App.

## Usage
This application is currently (Ver2.0) for Windows OS only.
This application offers a variety of options to customize your experience.

### 1. **Application Selection**
You can specify one application whose face you want to replace.  
Run the following command:
```shell
python main.py WINDOW_NAME
```
or
```shell
python main.py -window WINDOW_NAME
```
Please replace WINDOW_NAME with the name of the window you want to capture.  

## 2. Face Detection Model
You can specify the following face detection models:  

- YuNet: Relatively fast and strong in detecting small faces (By lowering -cth to around 0.6, it can detect almost all faces. However, false positives may increase, and illustrations may also be recognized as faces)
- face-detection-adas: Fast and strong in detecting slanted faces. It cannot detect small faces.
- face-detection-retail: Quite fast but slightly lower detection accuracy. It cannot detect small faces.
- Haar Cascade: A machine learning model included in OpenCV. The advantage over the above is that it works in various environments.

For the license of each model, please refer to the License section.

#### Face Detection Model Options
- ```--detection_model```, ```-dm``` : Path to the model above (YuNet is an onnx file, face-detection-adas, face-detection-retail is an xml file), specify 'haarcascades' for Haar Cascade. Default value: ```'./models/opencv_zoo/face_detection_yunet/face_detection_yunet_2023mar.onnx'```  
- ```--conf_threshold```, ```-cth``` (0~1): The higher the value, the more limited the detection to what can be recognized as a face. To detect blurred faces, you need to lower the value. Default value: ```0.6```  
- ```--nms_threshold``` (0~1): A parameter used to remove overlapping detection boxes using the NMS method. Only those with a confidence level higher than the specified value remain. Default value: ```0.3``` (YuNet only)  
- ```--top_k```: A parameter that specifies how many top detection boxes to use before NMS processing. The smaller the number, the faster it is. Default value: ```5000``` (YuNet only)
- `-device`: ['CPU', 'GPU', 'NPU'] Optimizes the OpenVINO model for the specified device. ('GPU' option does not support NVIDIA CUDA) Default value: ```'CPU'```

## 3. Face Recognition Model
You can specify the face recognition model.

- original: A well-known face recognition model 'dlib face recognition' with additional training for Asian faces has been accelerated by converting it to OpenVINO format.
- SFace: It is said to be a model with high recognition accuracy, but it may not be as good or even inferior for Japanese people. The speed is more than twice that of the dlib OpenVINO version.  

For the license of each model, please refer to the License section.

#### Face Recognition Model Options
- ```--recognition_model```, ```-rm```: Specify the path to the model above (SFace is an onnx file, dlib is an xml file). Default value: ```'./models/original/taguchi_face_recognition_resnet_v1_openvino_fp16_optimized.xml'```  
- ```--compare_face_method```, ```-cfm```: ['L2', 'COSINE'] L2 is the distance of feature vectors, COSINE is the cosine similarity of vectors. "L2" is recommended for original, and "COSINE" is recommended for SFace. Default value: ```'L2'```  
- ```--compare_face_tolerance```, ```-cft```: An indicator for judging whether faces are the same. For L2, 0 is a perfect match, and 0.4-0.6 is recommended. For COSINE, 1 is a perfect match, and 0.7-0.8 is recommended. Default value: ```0.4```  
- ```--face_alignment```, ```-fa```: ['Yes', 'No'] Processing to cut out faces detected by YuNet vertically. This process improves face recognition accuracy. There is almost no decrease in speed. Default value: ```'Yes'``` (YuNet-SFace combination only):
- `-device`: ['CPU', 'GPU', 'NPU'] Optimizes the OpenVINO model for the specified device. ('GPU' option does not support NVIDIA CUDA) Default value: ```'CPU'```

## 4. Avatars
Currently, this app only supports creating avatars using image files and does not support 3D models.  
Save your favorite avatar image in the 'avatars' folder.  
The image must be in jpg(jpeg), png(apng) or gif format (the extension must also be compatible).  
Avatars are assigned in alphabetical order of the names corresponding to the order in which faces are recognized.  
If there are not enough avatars, the last image will be assigned to the remaining faces.

### Avatar Options
- ```--specific-avatar-image```, ```-sai``` : Specify "image path" to replace all faces with the specified image.  
In this case, the face recognition model is not used, so it is slightly faster.

### Avatar License
Avatar images are created by generative AI. They are provided under the CC0 1.0 Universal License.

## 5. Installation
This app can be installed on Windows OS completely from scratch in 3 steps:

1. Install Python (version 3.12).
2. Download "Avatarian" as a zip file from this repository and unzip it.
3. Install the necessary files with ```pip install -r poetry-requirements.txt```

## 6. License
This application is made available under the MIT License. For the full license text, please refer to the LICENSE file included with this distribution.

### Licence of Original Model
- Model Name: **taguchi_face_recognition_resnet_v1_openvino_fp16_optimized**
- Format: OpenVINO (bin/xml)
- Size: 10.9MB
- License: This model is provided under the CC0 1.0 Universal License.
- Description: My Original face detection model is a OpenVINO version of **taguchi_face_recognition_resnet_model_v1.dat**, which is published under the CC0 1.0 Universal License in the [dlib-models](https://github.com/davisking/dlib-models/blob/master/README.md) repository. I have converted it to OpenVINO format and quantized to FP16.  
**taguchi_face_recognition_resnet_model_v1**, developed by Mr. Taguchi, is an improvement of the original **dlib_face_recognition_resnet_model_v1** and is characterized by its high accuracy in recognizing Asian faces.


### License of External Models
Each model stored in the models folder has its own license.

| Model Name | Format | Size | License | Source | Purpose |
|---|---|---|---|---|---|
| YuNet (Normal/INT8) | ONNX | 98KB/228KB | MIT | OpenCV_zoo/Copyright (c) 2020 Shiqi Yu | Face detection |
| face-detection-adas-0001 (PF16-INT8/PF16/FP32) | bin/xml | 1.6MB/2.3MB/4.2MB | Apache License 2.0 | Open Model Zoo/Copyright (c) 2023-2024 Intel Corporation | Face detection |
| face-detection-retail-0004 (PF16-INT8/PF16/FP32) | bin/xml | 0.8MB/1.3MB/2.3MB | Apache License 2.0 | Open Model Zoo/Copyright (c) 2020 Shiqi Yu | Face detection |
| face-detection-retail-0005 (PF16-INT8/PF16/FP32) | bin/xml | 1.5MB/2.2MB/4.0MB | Apache License 2.0 | Open Model Zoo/Copyright (c) 2023-2024 Intel Corporation | Face detection |
| SFace (Normal/INT8) | ONNX | 37MB/9.4MB | Apache License 2.0 | Open Model Zoo | Face recognition |

**For details on each license, please refer to the LICENSE file in the directory of each model.**






