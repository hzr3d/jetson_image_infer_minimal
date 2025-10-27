# 🧠 Jetson Orin Nano — Minimal Image Classification (No Camera)

A lightweight, **Jetson-friendly** image classification starter that runs directly on the Jetson Orin Nano (or other Jetson boards) **without TensorRT or PyCUDA**.  
It uses **OpenCV’s DNN module** to run ONNX models on **GPU (CUDA)** or CPU.

---

## 📘 Overview

This project demonstrates how to run **deep learning inference on images** using pretrained ONNX models (from the official ONNX Model Zoo) on NVIDIA Jetson boards.

It’s intentionally simple:
- No TensorRT engine building.
- No camera or live stream needed.
- Compatible with stock JetPack (Ubuntu + CUDA + cuDNN).
- Works fully offline once the models are downloaded.

### ✅ Supported models
- **SqueezeNet 1.0** (fast, ~5 MB)
- **ResNet-50 v1** (more accurate, ~97 MB)

Both are 1000-class ImageNet classifiers and come with label files.

---

## 🧩 Project structure

```
jetson_image_infer_minimal/
├── README.md                ← this file
├── run_image_cv2.py         ← main inference script
├── scripts/
│   └── download_models.sh   ← downloads ONNX models + labels
└── models/
    ├── squeezenet1.0-12.onnx
    ├── resnet50-v1-12.onnx
    └── imagenet_labels.txt
```

---

## ⚙️ Requirements

Make sure your Jetson is running **JetPack 5 or later** (includes CUDA, cuDNN, OpenCV).

Install these base packages:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy curl
```

> 💡 *Why apt and not pip?*  
> Jetson’s preinstalled OpenCV and NumPy are compiled together (against the same ABI).  
> Mixing pip’s NumPy (v2.x) breaks cv2 → use the apt versions.

If you prefer a virtual environment:
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```
*(Don’t install `numpy` or `opencv-python` via pip; they’re already provided by the system.)*

---

## 🚀 Setup & Run

### 1️⃣ Download the models
```bash
bash scripts/download_models.sh
```

### 2️⃣ Run inference on an image

Use any image (e.g., `~/Downloads/cat.jpg`).

#### SqueezeNet (faster, smaller)
```bash
python3 run_image_cv2.py   --image ~/Downloads/cat.jpg   --model models/squeezenet1.0-12.onnx   --topk 5
```

#### ResNet-50 (more accurate)
```bash
python3 run_image_cv2.py   --image ~/Downloads/cat.jpg   --model models/resnet50-v1-12.onnx   --topk 5
```

Expected output:
```
Inference time: 32.9 ms
Top predictions:
 1. tabby cat          prob=0.81
 2. tiger cat          prob=0.14
 3. Egyptian cat       prob=0.03
 4. lynx               prob=0.01
 5. Persian cat        prob=0.01
```

---

### 3️⃣ (Optional) Run on GPU

If your OpenCV was built with CUDA (it is, in recent JetPacks), add `--use-cuda`:

```bash
python3 run_image_cv2.py   --image ~/Downloads/cat.jpg   --model models/resnet50-v1-12.onnx   --topk 5 --use-cuda
```

You can confirm GPU activity with:
```bash
sudo tegrastats --interval 500
```

Look for **GR3D%** utilization rising while inference runs.

---

## 🧠 How it works

1. **Image preprocessing**
   - ResNet: PyTorch-style (224×224, RGB, mean/std normalization)
   - SqueezeNet: Caffe-style (227×227, BGR, mean = [104, 117, 123])

2. **Model loading**
   OpenCV’s `cv2.dnn.readNet()` reads the ONNX file and creates a backend network.

3. **Backend selection**
   - Default: CPU (`cv2.dnn.DNN_BACKEND_OPENCV`)
   - Optional: GPU (`cv2.dnn.DNN_BACKEND_CUDA`, `cv2.dnn.DNN_TARGET_CUDA_FP16`)

4. **Forward pass & output**
   The network produces logits for 1000 ImageNet classes, converted to softmax probabilities.  
   The script prints top-K predictions with human-readable labels.

---

## 🧩 Troubleshooting

| Problem | Cause | Fix |
|----------|--------|-----|
| `ImportError: numpy.core.multiarray failed to import` | Pip NumPy 2.x conflicts with OpenCV build | `sudo apt install python3-numpy` and uninstall pip’s NumPy |
| CUDA backend not found | OpenCV not compiled with CUDA | Use CPU mode or rebuild OpenCV with CUDA |
| Flat or random predictions | Wrong preprocessing | Use ResNet-50 or ensure SqueezeNet Caffe-style settings are active |
| `curl: (22) The requested URL returned error: 404` | Model Zoo URL changed | Use the validated `download_models.sh` script in this repo |

---

## 📊 Performance expectations (Jetson Orin Nano)

| Model | Backend | FP16 | Time (approx.) |
|--------|----------|------|----------------|
| SqueezeNet | CPU | – | 30–40 ms |
| SqueezeNet | GPU | FP16 | 10–20 ms |
| ResNet-50  | CPU | – | 150–200 ms |
| ResNet-50  | GPU | FP16 | 40–70 ms |

*(values depend on JetPack version and power mode)*

---

## 🧱 Next steps

- Swap in your own ONNX model (update `preprocess()` accordingly).
- Integrate a camera or RTSP stream for live video inference.
- Later, convert ONNX → TensorRT for maximum speed once you’re comfortable with the basics.

---

## 🧩 References

- [ONNX Model Zoo](https://github.com/onnx/models)
- [OpenCV DNN Documentation](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
- [NVIDIA Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-orin-nano-devkit)
- [Jetson Utilities (`jtop`)](https://github.com/rbonghi/jetson_stats)

---

### 🛠 Maintainer Notes

This repo intentionally avoids fragile TensorRT and PyCUDA dependencies for reliability on clean JetPack installations. Once your workflow is stable, you can branch it into a TensorRT-accelerated version for production.
