# Image Classification using Convolutional Neural Networks (CNN) with TensorFlow

This repository contains an example implementation of an image classification model using Convolutional Neural Networks (CNN) with TensorFlow. The model is trained to classify images into 10 different classes.

## Requirements

To run this code, you need to have the following dependencies installed:

- [ ] TensorFlow: A powerful machine learning library for building and training deep learning models.
- TensorFlow Keras: The high-level API for building and training neural networks with TensorFlow.
- TensorFlow Keras ImageDataGenerator: A utility for loading and preprocessing images for training the model.

You can install the required dependencies by running the following command:

```
pip install tensorflow
```

## Dataset

The dataset used for training and testing the model should be organized into two folders: `train` and `test`. Each folder should contain subfolders named after the class labels, with each subfolder containing the corresponding class images.

## Model Architecture

The CNN model architecture is defined using the TensorFlow Keras API. It consists of multiple convolutional and pooling layers followed by fully connected layers. The model architecture can be summarized as follows:

```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 222, 222, 32)      896
max_pooling2d (MaxPooling2D) (None, 111, 111, 32)      0
conv2d_1 (Conv2D)            (None, 109, 109, 64)      18496
max_pooling2d_1 (MaxPooling2 (None, 54, 54, 64)        0
conv2d_2 (Conv2D)            (None, 52, 52, 128)       73856
max_pooling2d_2 (MaxPooling2 (None, 26, 26, 128)       0
conv2d_3 (Conv2D)            (None, 24, 24, 256)       295168
max_pooling2d_3 (MaxPooling2 (None, 12, 12, 256)       0
flatten (Flatten)            (None, 36864)             0
dense (Dense)                (None, 512)               18874880
dense_1 (Dense)              (None, 10)                5130
=================================================================
Total params: 19,255,426
Trainable params: 19,255,426
Non-trainable params: 0
```

## Training and Evaluation

The model is trained using the `fit` method, which takes the training data generator and validation data generator as input. The number of epochs and batch size can be adjusted as needed. The model is compiled with an optimizer, loss function, and evaluation metric.

The `ImageDataGenerator` class is used for data augmentation and preprocessing. It performs data augmentation by applying random rotations, zooms, flips, and fills to the images, which helps in increasing the model's ability to generalize.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/ayam04/Tweet-Generator.git
   ```

2. Make sure you have the required dependencies installed.

3. Organize your dataset into `train` and `test` folders, with each folder containing subfolders for each class.

4. Modify the paths in the code to point to the correct dataset folders.

5. Run the code to train and evaluate the model.

Feel free to experiment with different parameters, architectures, and datasets to improve the model's performance.
