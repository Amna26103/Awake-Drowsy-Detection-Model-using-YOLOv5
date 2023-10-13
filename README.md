﻿# Awake-Drowsy-Detection-Model-using-YOLOv5
Certainly! Here's the updated README file with a warning about CUDA GPU and a suggestion to use Google Colab:

---

# Drowsiness Detection using YOLOv5

## Overview

This repository implements a robust Drowsiness Detection system using the YOLOv5 (You Only Look Once) object detection framework. The model is designed to accurately detect and classify instances of drowsiness in real-time video streams, making it particularly suitable for applications such as driver monitoring systems and other scenarios where fatigue detection is critical.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Model Evaluation](#model-evaluation)
- [Real-Time Detection](#real-time-detection)
- [Issues and Solutions](#issues-and-solutions)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contributing](#contributing)

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. **Organize Dataset:**
   - Prepare your dataset with labeled images and corresponding label files.
   - Organize the dataset into two folders: `images` for training images and `labels` for corresponding label files.

2. **Create Classes File:**
   - Create a `classes.txt` file listing all the classes present in your dataset.

3. **Configure Dataset YAML:**
   - Edit the `dataset.yaml` file to specify the dataset configuration, including paths to training and validation images, the number of classes, and class names.

## Training

Train the model using the following command:

```bash
python train.py --img-size 640 --batch-size 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt
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

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please check the [Contributing Guidelines](CONTRIBUTING.md) for more details on how to contribute to this project.

---

Feel free to adjust the content based on your specific preferences and requirements.
