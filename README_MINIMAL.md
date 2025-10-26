# Minimal Image Inference (OpenCV DNN) — Jetson-friendly
No TensorRT, no PyCUDA. Uses OpenCV's DNN (CUDA if available; falls back to CPU).

## Quickstart
```bash
# 1) Ensure OpenCV Python is installed (Jetson usually has it)
sudo apt update && sudo apt install -y python3-opencv curl

# 2) Get models
bash scripts/download_models.sh

# 3) Run inference on an image
python3 run_image_cv2.py --image /path/to/your.jpg --model models/squeezenet1.0-12.onnx --topk 5

# Optional: try ResNet50 (heavier, more accurate)
python3 run_image_cv2.py --image /path/to/your.jpg --model models/resnet50-v1-12.onnx --topk 5

# Force CUDA if your OpenCV build supports it (auto-detect is not guaranteed)
python3 run_image_cv2.py --image /path/to/your.jpg --model models/squeezenet1.0-12.onnx --use-cuda
```



What this project does

Goal: Run image classification locally on a Jetson Orin Nano without a camera.

How: Uses a tiny Python script to load an ONNX model and run inference with OpenCV DNN (CUDA if your OpenCV supports it, otherwise CPU).

Outputs: Top-K ImageNet predictions and inference time for any input image.

We also tried a TensorRT path, but to keep things clean and reliable on Jetson, the OpenCV DNN route is the recommended baseline. You can add TRT later once the baseline is working.

Run it (recommended minimal path)
0) Get the folder

If you haven’t already:

Download: jetson_image_infer_minimal.zip

Unzip anywhere, e.g. ~/image-infer

unzip jetson_image_infer_minimal.zip -d ~/image-infer
cd ~/image-infer

1) Install matching system packages

Make sure OpenCV and NumPy from apt are used together (they’re ABI-compatible on Jetson):

sudo apt update
sudo apt install -y python3-opencv python3-numpy curl
# Ensure pip's numpy (2.x) doesn't override the apt one:
python3 -m pip uninstall -y numpy || true


(If you prefer a venv, create it with --system-site-packages and don’t pip install numpy.)

2) Download the models + labels
bash scripts/download_models.sh
# You should see:
#   models/squeezenet1.0-12.onnx (~4.8 MB)
#   models/resnet50-v1-12.onnx   (~97  MB)
#   models/imagenet_labels.txt

3) Run inference on an image (no camera)

SqueezeNet (fastest; uses Caffe-style preprocessing in the script):

python3 run_image_cv2.py \
  --image /path/to/your.jpg \
  --model models/squeezenet1.0-12.onnx \
  --topk 5


ResNet-50 (more accurate):

python3 run_image_cv2.py \
  --image /path/to/your.jpg \
  --model models/resnet50-v1-12.onnx \
  --topk 5


(Optional) Try CUDA if your OpenCV supports it:

python3 run_image_cv2.py \
  --image /path/to/your.jpg \
  --model models/squeezenet1.0-12.onnx \
  --topk 5 --use-cuda

Troubleshooting (quick)

Nearly uniform predictions (all ~0.001): preprocessing mismatch → use ResNet-50 or ensure SqueezeNet branch (Caffe-style 227×227, BGR, mean=(104,117,123)) is active (already in the script).

NumPy/CV2 errors (NumPy 2.x conflict): remove pip NumPy and rely on python3-numpy from apt (1.x).

python3 - << 'PY'
import numpy as np, cv2, sys
print("python:", sys.executable)
print("numpy:", np.__version__, "->", np.__file__)
print("opencv:", cv2.__version__, "->", cv2.__file__)
PY


Expect NumPy 1.x and cv2 both under /usr/lib/python3/dist-packages.

Optional: Next steps

Swap in your own ONNX model (update input size/normalization in preprocess()).

Once baseline is solid, we can wire a TensorRT path for extra speed.

If you want, tell me which model you eventually plan to use (e.g., YOLO, ResNet variant), and I’ll tailor preprocess() + the command for it.
