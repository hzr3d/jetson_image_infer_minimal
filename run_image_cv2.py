#!/usr/bin/env python3
import argparse, os, sys, time
import cv2, numpy as np

def load_labels(path):
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            return [l.strip() for l in f]
    return None

def preprocess(img_bgr, model_name):
    name = os.path.basename(model_name).lower()

    # ---- SqueezeNet 1.0 (Caffe-style) ----
    if "squeezenet1.0" in name or ("squeezenet" in name and "1.0" in name):
        size = (227, 227)
        mean_bgr = (104.0, 117.0, 123.0)  # Caffe BGR mean
        # Caffe style: BGR, no std division, no /255 scalefactor
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, size),
            scalefactor=1.0,
            size=size,
            mean=mean_bgr,
            swapRB=False,   # stay in BGR
            crop=False,
        )
        return blob

    # ---- Default (PyTorch-style) for ResNet, etc. ----
    size = (224, 224)
    mean = (0.485*255, 0.456*255, 0.406*255)
    std  = (0.229, 0.224, 0.225)   # in 0..1 space
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, size),
        scalefactor=1.0/255.0,
        size=size,
        mean=mean,
        swapRB=True,
        crop=False,
    )
    # Divide by std per channel after swapRB (RGB)
    blob[:, 0, :, :] /= std[0]  # R
    blob[:, 1, :, :] /= std[1]  # G
    blob[:, 2, :, :] /= std[2]  # B
    return blob

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an image (jpg/png).")
    ap.add_argument("--model", required=True, help="Path to an ONNX model (squeezenet1.0-12.onnx or resnet50-v1-12.onnx).")
    ap.add_argument("--labels", default="models/imagenet_labels.txt", help="Optional labels file.")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--use-cuda", action="store_true", help="Try to run with CUDA backend if available.")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("Failed to read image")

    blob = preprocess(img, os.path.basename(args.model))

    net = cv2.dnn.readNet(args.model)

    if args.use_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except Exception as e:
            print("CUDA backend selection failed; falling back to CPU:", e)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    net.setInput(blob)
    t0 = time.time()
    out = net.forward()
    dt = (time.time() - t0) * 1000.0

    logits = out.reshape(-1)
    probs = softmax(logits)
    topk = min(args.topk, probs.shape[0])
    idxs = np.argsort(-probs)[:topk]

    labels = load_labels(args.labels)
    print(f"Inference time: {dt:.2f} ms")
    print("Top predictions:")
    for r, i in enumerate(idxs, 1):
        name = labels[i] if labels and i < len(labels) else f"class_{i}"
        print(f"{r:>2}. {name:>20s}  prob={probs[i]:.4f}")

if __name__ == "__main__":
    main()
