#!/usr/bin/env bash
set -euo pipefail
mkdir -p models

echo "Downloading SqueezeNet 1.0 (ONNX)…"
curl -L --fail --retry 5 --retry-delay 2 \
  -o models/squeezenet1.0-12.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx

echo "Downloading ResNet-50 v1 (ONNX)…"
curl -L --fail --retry 5 --retry-delay 2 \
  -o models/resnet50-v1-12.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx

echo "Downloading ImageNet labels…"
curl -L --fail --retry 5 --retry-delay 2 \
  -o models/imagenet_labels.txt \
  https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

echo "Done."

