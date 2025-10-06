Image Caption Generator

An AI model that automatically generates descriptive captions for images using Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) with LSTMs for sequential text generation.

🚀 Project Overview

This project aims to bridge the gap between computer vision and natural language processing by generating meaningful captions for input images.

CNN: Used to extract image features (acts as the visual encoder).

RNN/LSTM: Used to decode image features into words (acts as the language model).

The number of LSTM layers equals the number of words in the target caption, allowing the model to learn temporal word dependencies dynamically.

🧠 Model Architecture

1. CNN (Encoder):

Pre-trained CNN (e.g., VGG16, InceptionV3, or ResNet50)

Removes the top layer to extract high-level visual features

Outputs a feature vector representing the image

2. RNN/LSTM (Decoder):

Takes the encoded image vector as input

Generates captions word by word

The model’s depth (number of LSTM layers) is set equal to the number of words in the generated caption sequence

3. Training:

Image features + corresponding captions are trained together

Cross-entropy loss used for caption prediction

Word embeddings (like GloVe) are used to represent words in dense vector space

🧩 Technologies Used
Component	Technology
Language	Python
Deep Learning Framework	TensorFlow / Keras
Image Processing	CNN (VGG16 / InceptionV3 / ResNet)
Text Processing	RNN / LSTM
Dataset	Flickr8k / Flickr30k / MS COCO
Visualization	Matplotlib, Seaborn
Environment	Jupyter Notebook / Google Colab
