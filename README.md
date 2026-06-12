# ESPCN Video Super Resolution

A lightweight deep learning project for enhancing low-resolution videos using ESPCN (Efficient Sub-Pixel Convolutional Neural Network) with PyTorch and OpenCV.

The system extracts video frames, upscales them using a trained ESPCN model, then reconstructs them back into an enhanced video.

---

# Features

* ESPCN implementation using PyTorch
* CUDA GPU acceleration
* Video frame extraction + reconstruction
* Bicubic vs ESPCN comparison
* PSNR / SSIM evaluation
* Data augmentation during training
* Modular and reproducible pipeline

---

# Pipeline

```text id="pipelinefinal"
Input Video
    ↓
Frame Extraction
    ↓
Bicubic Downscaling
    ↓
ESPCN Inference
    ↓
Enhanced Frames
    ↓
Video Reconstruction
    ↓
Comparison Video
```

The comparison video shows Bicubic interpolation and ESPCN output side-by-side for visual evaluation.

---

# Project Structure

```bash id="structurefinal"
super-res-for-video/
│
├── dataset.py
├── model.py
├── train.py
├── video_inference.py
├── extract_frames.py
├── reconstruct_video.py
├── compare_video.py
├── metrics.py
├── requirements.txt
│
├── dataset/
├── input/
├── output/
├── video_frames/
└── enhanced_frames/
```

---

# Installation

## Clone repository

```bash id="clonefinal"
git clone <repository-url>
cd super-res-for-video
```

---

## Create virtual environment

### Windows

```bash id="venvfinal"
python -m venv venv
venv\Scripts\activate
```

### Linux / MacOS

```bash id="venvlinuxfinal"
python3 -m venv venv
source venv/bin/activate
```

---

## Install dependencies

```bash id="installfinal"
pip install -r requirements.txt
```

---

# Training

You may opt to use our trained model or train the model yourself with our preset settings and dataset
Place high-resolution images inside:

```bash id="datasetfinal"
dataset/hr/
```

Run training:

```bash id="trainfinal"
python train.py
```

Current training setup:

| Parameter     | Value    |
| ------------- | -------- |
| Epochs        | 200      |
| Batch Size    | 16       |
| Optimizer     | Adam     |
| Loss Function | MSE Loss |
| Scale Factor  | 4×       |

The trained model will be saved as:

```bash id="modelsavefinal"
espcn.pth
```

---

# Running Video Super Resolution

Make sure you are on the correct direction to the folder: ...\super-res-for-video

Run virtual environment (ignore this if you already activated)
### Windows

```bash id="venvfinal"
venv\Scripts\activate
```

### Linux / MacOS

```bash id="venvlinuxfinal"
source venv/bin/activate
```
Then run
```
python app.py
```

# Evaluation

Run metrics:

```bash id="metricsfinal"
python metrics.py
```

Current results:

| Metric | Value  |
| ------ | ------ |
| PSNR   | ~37.88 |
| SSIM   | ~0.989 |

Observations:

* ESPCN produces sharper edges than Bicubic interpolation
* Improvements are more noticeable when zoomed in
* CUDA significantly improves inference speed

---

# GPU Support

The project automatically uses CUDA if available.

Check CUDA:

```bash id="checkcudafinal"
python -c "import torch; print(torch.cuda.is_available())"
```

Tested on:

* NVIDIA RTX 3060

---

# Common Issues

## OpenCV missing

```bash id="opencverrorfinal"
ModuleNotFoundError: No module named 'cv2'
```

Fix:

```bash id="opencvfixfinal"
pip install opencv-python
```

---

## Black output video

Usually caused by:

* Corrupted frames
* Codec mismatch
* Invalid frame sizes

Fix:

* Delete generated frames
* Re-run inference
* Reconstruct video again

## Slow inference speed

CPU inference can be very slow for longer videos.

Recommended:

* Use CUDA GPU acceleration
* Reduce video resolution
* Use shorter test videos during development
---

# Future Improvements

* ESRGAN implementation
* Temporal consistency between frames
* Real-time webcam super resolution
* GUI/Desktop application
* Larger training dataset
* Better perceptual loss functions

---

# Tech Stack

* Python
* PyTorch
* OpenCV
* Pillow
* NumPy

---
# References
* Shi et al. — Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
* PyTorch Documentation
* OpenCV Documentation
---
# Authors

* Phạm Trần Tuấn Minh
* Nguyễn Quang Hải Anh
