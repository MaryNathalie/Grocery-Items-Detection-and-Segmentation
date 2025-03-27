# Grocery Items Detection and Segmentation

<p align="center">
<img src="https://github.com/MaryNathalie/Grocery-Items-Detection-and-Segmentation/blob/main/images/sample.png" width=60% height=60%>
</p> 

This repository contains the implementation of a YOLO-based object detection and segmentation model for grocery items. The project focuses on improving model accuracy and real-time inference speed, with deployment via a web-based application using Gradio and WebRTC.

🔗 [1st Iteration Presentation](https://github.com/MaryNathalie/Grocery-Items-Detection-and-Segmentation/blob/main/documents/presentation_01.pdf) | [2nd Iteration Presentation](https://github.com/MaryNathalie/Grocery-Items-Detection-and-Segmentation/blob/main/documents/presentation_02.pdf)

### 📌 Features
- COCO-formatted **Dataset Preparation** into YOLO format, ensuring bounding boxes and segmentations remain within image boundaries.
- **Object Detection & Segmentation** using YOLO.
- **Dataset Augmentation** via Albumentations.
- **Experiment Tracking** with CometML.
- **Live Webcam Inference** powered by WebRTC and Twilio API.
- **Web-Based UI** using Gradio.
- **Optimized Training Pipeline** with hyperparameter tuning.

### 📊 Dataset Summary

The dataset consists of images of grocery items. Data augmentation techniques such as rotation, translation, shear, and mixup were applied to improve model robustness.

<div align="center">
  
| #  | Class              | Training Images | Validation Images | Specific Brand                               |
|----|--------------------|-----------------|-------------------|----------------------------------------------|
| 1  | Bottled Soda       | 477             | 58                | Coca-Cola (Coke Zero)                        |
| 2  | Cheese             | 310             | 40                | Eden (Classic)                               |
| 3  | Chocolate          | 459             | 59                | KitKat (Chocolate)                           |
| 4  | Coffee             | 404             | 41                | Nescafe Original (Classic)                   |
| 5  | Condensed Milk     | 370             | 46                | Alaska (Classic)                             |
| 6  | Cooking Oil        | 467             | 55                | Simply Canola Oil                            |
| 7  | Corned Beef        | 442             | 58                | Purefoods (Classic, Spicy)                   |
| 8  | Garlic             | 317             | 33                | Whole                                        |
| 9  | Instant Noodles    | 431             | 42                | Lucky Me! (Sweet and Spicy)                  |
| 10 | Ketchup            | 477             | 47                | UFC (Banana)                                 |
| 11 | Lemon              | 324             | 38                | Whole                                        |
| 12 | All-purpose Cream  | 451             | 49                | Nestle (Classic)                             |
| 13 | Mayonnaise         | 319             | 31                | Lady's Choice (Classic)                      |
| 14 | Peanut Butter      | 485             | 35                | Lady's Choice, Skippy                        |
| 15 | Pasta              | 443             | 57                | Royal Linguine                               |
| 16 | Pineapple Juice    | 449             | 50                | Del Monte (Fiber, ACE)                       |
| 17 | Crackers           | 462             | 47                | Skyflakes, Rebisco                           |
| 18 | Sardines (Canned)  | 305             | 45                | 555 (Tomato)                                 |
| 19 | Pink Shampoo       | 444             | 56                | Sunsilk (Smooth and Manageable)              |
| 20 | Soap               | 446             | 54                | Dove (Lavender)                              |
| 21 | Soy Sauce          | 452             | 48                | Silverswan                                   |
| 22 | Toothpaste         | 456             | 44                | Colgate (Advanced White)                     |
| 23 | Canned Tuna        | 461             | 61                | Century Tuna (Original, Hot and Spicy)       |
| 24 | Alcohol            | 426             | 34                | Green Cross (Ethyl)                          |

</div> 

For more information about the dataset, email me at marynathaliedelacruz@gmail.com

### 🏗 Model Training

The project initially used the YOLO11-Nano model but later upgraded to YOLO11-Medium for better accuracy. The training pipeline is configured as follows:

```python
model.train(
  data="dataset.yaml",
  project="MEX6_runners",
  epochs=200,
  imgsz=640,
  batch=16,
  patience=10,
  name="run02",
  degrees=30,
  hsv_v=0.3,
  translate=0.4,
  shear=0.3,
  flipud=0.05,
  mixup=0.4,
  copy_paste=0.3
)
```

### 🔧 Data Augmentations

- HSV_V (0.3): Adjust brightness for lighting variations.
- Translate (0.4): Detect partially visible objects.
- Degrees (30): Rotate images to recognize orientations.
- Shear (0.3): Simulate different viewing angles.
- FlipUD (0.05): Introduce variability without distorting objects.
- Mixup (0.4): Blend images for better generalization.
- Copy-Paste (0.3): Increase object instances and occlusion.

### 🌍 Web-Based Deployment

- WebRTC Configuration
  - Uses Twilio API Tokens to configure ICE servers for smooth real-time webcam streaming.
  - Establishes peer-to-peer (P2P) connections for low-latency inference.
- Gradio Interface
  - WebRTC Stream for real-time detection.
  - Confidence Threshold Slider for adjustable detection sensitivity.

### 🚀 Deploying the Gradio App

Deploy the Gradio interface with app.py, follow these steps:

1. Clone this repository

```python
git clone 
```

2. Install dependencies:

```python
pip install -r requirements.txt
```

3. Run the application:

```python
python app.py
```

4. Access the web interface. If deploying on a remote server, use share=True in gr.Interface.launch() to get a public link.

### 📊 Results

Comparing models YOLO-M with data augmentation (blue) and YOLO11-L without additional data augmentation (pink). Figures are from wandb:

- Reduced validation box loss, class loss, segmentation loss, and DF1 loss.

<p align="center">
<img src="https://github.com/MaryNathalie/Grocery-Items-Detection-and-Segmentation/blob/main/images/validation.png" width=50% height=50%>
</p> 

- Increased computational efficiency with optimized hyperparameters.

<p align="center">
<img src="https://github.com/MaryNathalie/Grocery-Items-Detection-and-Segmentation/blob/main/images/parameters.png" width=50% height=50%>
</p> 

- Improved classification accuracy.

| Model Type                                          | Precision | Recall | mAP50 | mAP95 |
|-----------------------------------------------------|-----------|--------|-------|-------|
| YOLO-M with data augmentation                       | **96.9**      | 92.7   | **95.6**  | **88.9**  |
| YOLO11-L without additional data augmentation       | 96.0      | **92.9**   | 95.5  | 88.7  |

- Improved classification accuracy.

| Model Type                                          | Precision | Recall | mAP50 | mAP95 |
|-----------------------------------------------------|-----------|--------|-------|-------|
| YOLO-M with data augmentation                       | **96.6**      | **92.2**   | **94.8**  | **84.9**  |
| YOLO11-L without additional data augmentation       | 96.9      | 92.1   | 94.7  | 84.8  |

### 📜 Future Work

- Deploy ONNX inference with onnxruntime.InferenceSession().
- Perform hyperparameter tuning with Ray Tune.
- Experiment with YOLO11-X for enhanced accuracy.
- Collect more real-world images to improve generalization.
- Implement frame aggregation for smoother predictions.
