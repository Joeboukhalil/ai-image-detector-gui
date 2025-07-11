# AI vs Real Image Detector (GUI with Keras)

This is a lightweight Convolutional Neural Network (CNN) model built with TensorFlow/Keras that detects whether an image is AI-generated (fake) or a real photo. The model is trained on 120,000+ images and provides predictions through a simple GUI made with Tkinter.

## Features
- Detects AI vs real images from local files
- Easy-to-use graphical interface
- Trained on CIFAKE dataset (CIFAR-10 real images vs AI-generated images)

## Requirements
- Python 3.10 (recommended)
- TensorFlow 2.10
- Pillow

## Getting Started

1. Clone this repo:
https://github.com/Joeboukhalil/ai-image-detector-gui.git


2. Run the GUI:
python gui_detector.py


3. Browse an image and see if it's **Fake (AI-generated)** or **Real**!

## Dataset Source
Trained on the CIFAKE dataset:
https://www.kaggle.com/datasets/jjblanchard/cifake-real-and-ai-generated-synthetic-images

## Transparency
This model was built with the help of AI (ChatGPT) for educational purposes.


## requirements.txt
tensorflow==2.10.0
pillow

