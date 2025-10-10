Garbage classifier
CNNs, Transfer Learning & Model Optimization – Assignment

Overview

This assignment demonstrates the training of CNN models over a garbage image dataset, using yolo model and optimization of the best model into ONNX and TensorRT.

It is divided into four parts:

1. Part A – Train a CNN from scratch
2. Part B – Fine-tune a pre-trained model
3. Part C – Apply a pre-trained YOLO model for a creative use case
4. Part D – Optimize and benchmark with TensorRT

Dataset: [Garbage Classification Dataset (6 classes)](https://drive.google.com/file/d/1nmqD6P14FvoMqqmIseqkuIe2Y40iM2SG/view)

Requirements

* Python 3.8+
* PyTorch, Torchvision
* PyTorch Lightning (optional)
* Weights & Biases (wandb)
* OpenCV, Matplotlib, Seaborn
* ONNX, TensorRT (for optimization)

Install dependencies:

```bash
pip install torch torchvision pytorch-lightning wandb onnx onnxruntime opencv-python matplotlib seaborn
```

1. Model Design

Custom CNN with 5 convolutional blocks:
  Conv → Activation → MaxPooling
1 Fully connected (dense) layer + output layer with 6 neurons.


2. Theoretical Analysis

Total computations in the network: 
Formula: Output_Size x k^2 x input_channel x output_channel x 2

for,
conv layer 1: 5,66,23,104
conv layer 2: 15,09,94,944
conv layer 3: 15,09,94,944
conv layer 4: 15,09,94,944
conv layer 5: 15,09,94,944
fully connected layer: 1,96,608

total= 66,07,99,488 computations

Total number of parameters:
Formula: (Kernel_width x kernel_height x input_channel + 1) x output_channel
for,
conv layer 1: 448
conv layer 2: 4,640
conv layer 3: 18,496
conv layer 4: 73,856
conv layer 5: 2,95,168
Total= 3,92,608

Fully connected layer: (input x output) + output
(256x8x8 x 6) + 6 = 98310

Total computations in network = 490918 parameters

3. Training Setup

Dataset split: 60% train, 20% validation, 20% test.
Data augmentations: RandomHorizontalFlip() and RandomRotation()
Optimizer & learning rate schedule: Adam optimizer with LR: 0.001, 0.0001, 0.0005

4. Hyperparameter Sweep

Hyperparameters explored:

-Filters: [32, 64, …]
-Activations: ReLU
-Filter organization: doubling
-Batch norm: No
-Dropout: 0,0.5
-Data augmentation: Yes [RandomHorizontalFlip() and RandomRotation()]

Observations: I tried using different batch sizes and 32 worked the best because with 16 the model was learning even the useless details and with 64 it was leaving out important details (like if the bottle is green it would detect glass, even if its plastic). I tried different learning rates and 0.001 was stable and increased accuracy and using a dropout of 0.5 prevented overfitting. Adding data augmentations like flipping the images and rotating them increased the accuracy by 3%.

5. Results

Best hyperparameters: 
Filter Organization: Doubling (16 → 32 → 64 → 128 → 256)
Filters per Layer: [16, 32, 64, 128, 256]
Activation Function: ReLU
Dropout: 0.5
Data Augmentation: Horizontal Flip + Random Rotation
Optimizer: Adam
Learning Rate: 0.001
Batch Size: 32
Number of Epochs: 25

Test accuracy: 74%

Part B – Fine-tuning a Pre-trained Model

1. Model Used

-Pre-trained model: ResNet50
-Final layer replaced with 6 neurons.

2. Fine-tuning Strategies
Tried:
1. Freeze all except last layer
2. Freeze first 40 layers, train the rest
3. Freezing 1st conv layer and consequent 2 residual layer with batch norm layer 1 and training on the rest.

3. Training
-Same train/val/test split as Part A.

4. Results & Observations

Final test accuracy 86%

Observations:I tried freezing only the final layer and it worked best for my small dataset because the pretrained layers already captured useful features, so the model didn’t overfit. Freezing the first 40 layers gave slightly lower accuracy and more fluctuations because training some early layers with little data made it unstable. Freezing by blocks (conv1 to layer2) performed the worst since too many layers were trainable for a small dataset, causing overfitting and unstable validation accuracy.

Which strategy worked best?
-Freezing all the layers except last worked the best giving 86.4 % accuracy.
Comparison with Part A (scratch training)
-Significantly performed better by increasing the accuracy from 74% to 86%

Part C – YOLO Pre-trained Model Application

1. Creative Use Case
-Task: crowd management 
-Model: YOLOv8 
Demo video link: https://www.youtube.com/watch?v=Hp3uEgkls9E
Visualizer video: https://drive.google.com/file/d/1BT7SHggHgcI3ss08b5kW1Gp6BwLsSZvz/view?usp=sharing


Part D – TensorRT Conversion & Benchmarking

1. Export & Conversion

-Model exported to ONNX.
-ONNX → TensorRT engine built.

2. Benchmarking

Inference speed comparison:

-PyTorch CPU: 0.0292s/img, 34.26 img/s
-PyTorch GPU: 0.0009s/img, 1092.27 img/s
-TensorRT Model: 0.0055s/img, 181.39 img/s

3. Results & Observations

Speedup factor: GPU is approx 32× faster than CPU. TensorRT is approx 6× faster than CPU, but approx 6× slower than GPU in FP32 mode.(might overtake if used other modes)
Accuracy changes: none
Quantization effects: none
Deployment challenges: 
-The session kept crashing due to CUDA/TensorRT setup issues.
-The ONNX model only accepted batch size 1, so I had to adjust inputs instead of running larger batches.
-trtexec didn’t run properly in Colab, so I used ONNX Runtime with TensorRT/CUDA providers instead.
