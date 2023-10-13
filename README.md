# Awake-Drowsy-Detection-Model-using-YOLOv5

## Overview

This repository implements a Drowsiness Detection system using the YOLOv5 (You Only Look Once) object detection framework. The model is designed to accurately detect and classify instances of drowsiness in real-time video streams, making it particularly suitable for applications such as driver monitoring systems and other scenarios where fatigue detection is critical.

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Model Evaluation](#model-evaluation)
- [Real-Time Detection](#real-time-detection)
- [Issues and Solutions](#issues-and-solutions)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

## Installation

### Clone the Repository

```bash
git clone https://github.com/Amna26103/Awake-Drowsy-Detection-Model-using-YOLOv5
cd your-repository
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Training
**Warning: Ensure CUDA GPU Availability for Faster Training and Inference.Otherwise use Google Colab**
 
Train the model using the following command:

```bash
python train.py --img-size 2330 --batch-size 3 --epochs 500 --data dataset.yaml --weights yolov5s.pt --workers 2
```

**Note: Ensure that you have a CUDA-enabled GPU for faster training.**

If you don't have a GPU, consider using Google Colab for cloud-based GPU acceleration.

## Model Evaluation

View training results in real-time using the `yolov5/runs/train/expX` folder, where `X` corresponds to the experiment number. Evaluate model performance using metrics provided in the training summary.

## Real-Time Detection

Load the trained model for real-time detection:

```python
import torch

model = torch.hub.load('ultralytics/yolov5:v5.0', 'custom', path='yolov5/runs/train/expX/weights/best.pt')
results = model('data/test.jpg')
results.show()
```

Replace `X` with the experiment number of the trained model. This allows real-time detection on new images.

## Issues and Solutions

- **Broken Pipe Error:**
  If encountering a "broken pipe" error during training, use the `--workers` flag:

  ```bash
  python train.py --img-size 640 --batch-size 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt --workers 2
  ```

- **Model Reload Issue:**
  For issues reloading the model, add `--force_reload true` during model loading.

**Warning: Ensure CUDA GPU Availability for Faster Training and Inference.**

## Future Improvements

Possible enhancements to the model include feature extraction from eyes and mouth for additional cues on drowsiness. This can provide more granular information about the state of the driver or individual.

## Acknowledgments

This project is built upon the YOLOv5 framework. Special thanks to the contributors of the YOLOv5 repository for their continuous efforts in developing and maintaining this powerful object detection framework.
