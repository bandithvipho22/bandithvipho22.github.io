---
title: "Cambodia Sign Language Recognition and Translation Text-To-Speech"
excerpt: "<br/><img src='/images/ksl_proj.png'>"
collection: portfolio
---

![Sign Language](/images/ksl_proj_t2.png)

---

# I. Introduction
Sign Language Processing (SLP) is a field of research that focuses on the development of technologies that can understand and generate sign language, with the rise of Artificial Intelligence, deep learning and computer vision technologies, the field of SLP has seen significant advancements in recent years. 

![Skeleton](/images/skeleton.png)

---

## 1.1. Objective

- The primary objective of this project is to develop a Cambodia Sign Recognition Model by using deep learning and deploy on web application for Cambodia Sign Language Translation, which is using real-time camera to recognize Cambodia sign languages gesture and Translate it to text and Khmer Spoken.

![demo](/images/demo.png)

- At the initial stage: the application begins with web application which allows users to perform sign language in front of the camera attached to and the webcam of laptops. The Recognition model can recognition 21 signs (classes).

## 1.2. Scope of work
- Data Collection (Currently we got 21 signs)
- Training Model (Used LSTM)
- Evaluate our Model
- Real-Time Inference (Check speed and Accuracy)
- Integrated with Text-To-Speech Model ( used TTS APIs)
- Deploy in web application (Used Flask Frameworks)

---

# II. Methodology
In this section, we describe the technical pipelines of the recognition model and text to speech translation. In-depth details on the data collection, selecting model, model training and inference stages for web application deployments.

## 2.1. Data Collection
The data collection process involves capturing sequential data from video streams. We utilized the Media Pipe library to extract hand landmarks from each frame in the video. These landmarks represent key points on the hands, allowing us to focus on relevant features for sign recognition. The extracted landmark data is then saved in a structured array format (.npy files), which is efficient for storage and training purposes.

![data_collection](/images/data_collection.png)

To ensure the model performs well across a variety of conditions, data is collected from
multiple participants, each performing gestures in different environments, lighting conditions,
and camera angles. Each gesture is captured in sequences of 30 frames, stored as .npy files for
efficient processing.

![data_collection_v2](/images/data_collection_v2.png)

As shown in image above, the dataset includes sign language gestures for words such as
['hello', 'I', 'go', 'home', 'like', 'today']. The input data is processed as sequential data, where each
gesture is represented by a sequence of frames. We then perform feature extraction to convert
raw images into key-points, capturing critical information about hand, pose, and face landmarks. To enhance the dataset and improve model generalization during training, we apply
data augmentation techniques such as flipping, rotation, and adding noise. These
augmentations help create a more robust model capable of handling variations in gestures,
camera angles, and lighting conditions. Finally, all the data that is collected we store it in file
.npy as the array form.

## 2.2. Sign Recognition Model
We trained our Cambodia Sign Recognition Model by using sequence model. In this research I used **LSTM** for training my sign recognition model that extracted keypoints from MediaPipe (such as hand and pose landmarks). The network learns to identify patterns in these sequences, enabling it to classify gestures like (“Hello”, “Home”, or “Go”, ...).

![LSTM](/images/LSTM.png)

Below, we outline the architectural setup and the key components of the model, the model comprises 8 layers, each serving a specific purpose in the learning proces.

![LSTM](/images/LSTM_t2.png)

Here, the following code defines the architecture of the LSTM model with Tensorflow platform that I use for training model:

```python
def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh', input_shape=input_shape),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        LSTM(256, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

## 2.3. Text-To-Speech Generation
In the Text-to-Speech (TTS) generation module, is designed to convert the text output
from the sign language recognition model into spoken Khmer, facilitating real-time
communication. The process follows a series of steps to ensure high-quality speech generation.

![TTS](/images/TTS.png)

As we can see in Figure above, the Text-to-Speech (TTS) system converts the recognized sign
language gestures into Khmer spoken, enabling real-time communication. The model used in
this project is the **“facebook/mms-tts-khm”** APIs from VITS model, which is designed for the
Khmer language. The process involves several key steps such as:

![TTS](/images/TTS_2.png)

- Recognition Model: After recognizing a Sign Language from gesture, the system
predicted the detected sign into Khmer Text using a predefined dictionary.
- Generate Audio: Once the recognition model predicted the sign languages to output
label (Text), the system uses the VITS model to generate the corresponding audio
waveform. The model tokenizes the Khmer text and processes it through the neural
network to produce a waveform of the speech.
- Generate waveform: Converted into a compatible audio format using the **PyDub**
library. 

To use this VITs Model for Text-To-Speech Generation, we can install the latest version of the
library by following command below:

```
pip install --upgrade transformers accelerate
```

Then, we can run inference with TTS APIs by following code-snippet from Hugging Face:

```python
from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-khm")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-khm")

text = "some example text in the Khmer language"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():

output = model(**inputs).waveform
```

*** Noted that ***: The output result of waveform can be saved as a .wav file. To integrate with Khmer sign
recognition system, we need convert and scale it into 16-bit PCM format, which is suitable
for playback as audio. Then use PyDub to playback the audio as a Khmer spoken.

## 2.4. Experiments Setting
The training process is optimized with the following hyperparameters to enhance model
performance and generalization:

![Parameter](/images/parameters.png)

This model is trained with Categorical Crossentropy and Adam Optimizer, making it
suitable for multi-class classification tasks like sign language recognition.