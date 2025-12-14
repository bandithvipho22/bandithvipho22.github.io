---
title: "Cambodia Sign Language Recognition and Translation Text-To-Speech"
excerpt: "<br/><img src='/images/ksl_proj.png'>"
collection: portfolio-2
---

![Sign Language](/images/ksl_proj_t2.png)

# Table of Contents
- [I. Introduction](#i-introduction)
- [II. Methodology](#ii-methodology)
  - [2.1. Data Collection](#21-data-collection)
  - [2.2. Sign Recognition Model](#22-sign-recognition-model)
  - [2.3. Text-To-Speech Generation](#23-text-to-speech-generation)
  - [2.4. Interface](#24-interface)
    - [2.4.1. Server](#241-server)
    - [2.4.2. Client](#242-client)
- [III. Result and Discussion](#iii-result-and-discussion)
- [IV. Achievements](#iv-achievements)
- [V. Our Goals](#v-our-goals)

---

***Notice***: For full technical details, architecture design, and experimental results, please refer to the complete report:
[[View my Report](/files/KSL_Project.pdf)] and my demo video [[Demo videos](https://drive.google.com/drive/folders/13BYBKfAXmTQ-JzGwgc5eopT5qk_auznO?usp=drive_link)].

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

**Noted that**: The output result of waveform can be saved as a .wav file. To integrate with Khmer sign
recognition system, we need convert and scale it into 16-bit PCM format, which is suitable
for playback as audio. Then use PyDub to playback the audio as a Khmer spoken.

## 2.4. Interface
The interface of the system is designed to provide seamless interaction between the
client and server components, enabling real-time Khmer sign language recognition, translation,
and text-to-speech generation. 
- The server provides a modular implementation of sign recognition and text-to-speech generation that is accessible via a **WebSocket**. 
- The client is a web application that serves as user-friendly interface for interacting with the two components which is **sign recognition** and **text-to-speech** translation.

## 2.4.1. Server
The server is implemented in Python using the Flask web framework. It provides a
WebSocket API that allows clients to interact with the sign recognition and text-to-speech
Translation components. The server captures video streams, processes the frame using
the recognition model, and emits the sign letter or text to the client through the WebSocket
messages. This architecture allows for real-time sign recognition.

![server](/images/server.png)

This flowchart demonstrates the step-by-step process followed by the server to process input, predict signs, update sentences, and provide feedback to the user.
- **Start and Initialization**: The system begins its operation when the server is launched. The system wait for input frame to process Sign Recognition Model and make prediction.
- **User Input**: There are frame from the client’s camera for processing. A **Clear** command to reset the system's data and a **Stop** command to end the server's operation.
- **Frame Processing**: When the frame is received, the system starts analyzing the
incoming video frame. Then, extract landmarks by using **MediaPipe** Library. After that,
store the extracted landmarks in a buffer to construct a sequence. The buffer helps in
capturing the context of movements over time.
- **Sequence length check**: The system checks whether the buffer contains 30 frames (the
required sequence length). If **“No”**, the system returns to waiting for more frames, as a
complete sequence is needed for accurate predictions. If **“Yes”**, the system proceeds to
the Prediction step.
- **Prediction and Update sentence**: Once a full sequence of 30 frames is collected, the predicted sign is appended to the current sentence. If the same word is
repeated unnecessarily, the system avoids adding it to maintain sentence clarity.
This step ensures the server constructs meaningful sentences from the
recognized signs.
- **Text-To-Speech Translation**: The system evaluates whether the updated sentence
needs to be spoken aloud using TTS APIs, If **“No”**, the sequence buffer is cleared, and the system waits for new input, If **“Yes”**, the server converts the sentence into speech, plays it back to the user,
and then clears the buffer.

## 2.4.2. Client
For client I used opensource template and I modified them for my task. The client is a web application front-end developed using **Next.js**, a **React-based** framework, to manage both the front-end interface and server-side functionality.
- TypeScript was used to write the front-end code, with “. tsx” files defining the user interface components.
- The client establishes a connection via **WebSocket**, to enable interaction with the server that's allowing real-time communication with the recognition and text-to-speech translation
components.
- The client displays a live webcam feed for sign recognition, overlaying
recognized text directly on the stream. Additionally, transcribed words from the server,
generated by the sign recognition model, are displayed on the right side of the interface in real
time, providing an intuitive and dynamic user experience as shown in **Figure below**.

![Sign Language](/images/ksl_proj_t2.png)

# III. Result and Discussion

**Experiments Setting**:

The training process is optimized with the following hyperparameters to enhance model
performance and generalization:

![Parameter](/images/parameters.png)

This model is trained with **Categorical Crossentropy** and **Adam Optimizer**, making it
suitable for multi-class classification tasks like sign language recognition.

The Cambodia sign language recognition model was trained using a TUF GTX 1650 Ti
graphics card with 4GB VRAM, involved a large dataset with 5,948 augmented sequences and
involved splitting into **“training set”** and **“validation sets”**. 

![data load](/images/data_load.png)

In this study, we loaded a total of 12,552 sequences, each comprising 30 frames. The dataset includes 21 unique classes, with a relatively balanced distribution across each class. To enhance the robustness of the
model, data augmentation was performed, increasing the total number of sequences to 18,818.

**Accuracy and Loss**:

The graphs you provided show the accuracy and loss of the model over epochs during training
and validation:
- Training Accuracy: Starts around 0.5 (50%) and increases steadily over epoch. Reaches
close to 1.0 (100%) by the end of training, indicating the model is learning well on the
training.
- Validation Accuracy: Starts similarly around 0.5 but increases more slowly. Plateaus
around 0.8 (80%) by the end of training, which is lower than the training accuracy.

![data accuracy](/images/accuracy.png)

**Output layers and parameters**:

The model consists of a total of 511,169 parameters, which makes it relatively lightweight
compared to larger deep learning architectures. The model processes 1,000 samples in just 1.65
seconds, which is extremely fast. It can handle high-throughput scenarios where multiple
samples need to be processed quickly.

![parameter](/images/parameters_lstm.png)

**Output result deployed on web application**:

The interface is designed to provide real-time sign language recognition and translation. 
- As shown in the web interface, the right is real-time video steaming and it recognized the predicted sign which shown in the lift side.

![ksl result](/images/ksl_result.png)

**Discussion**:

As the result, this research optain Cambodia Sign Recognition Model which can recognize 21 signs and convert each sign into speech by using TTS ViTs pre-trained model APIs. In this work is just a initial stage and also leave some challenges and limitations such as:
- Difficult to collect the Cambodia Sign Language Dataset
- For this first stage, the dataset we collect by myself with my teammate so some action will be different from the deaf and mute people that could led our dataset not generalize.
- When recognizing complex gestures, fast hand movements, or occlusions, which
could hinder its ability to maintain consistent recognition accuracy.

# IV. Archivements
We used our Cambodia Sign Recognition and Translation project to join several competition to learn and explore more Idea to fulfill what we have missed.

**1. Business Model Competition, Cambodia (BMC) in march, 2025**:

- Digital Tech Award, Trip to San Francisco, United States of America.

![arch01](/images/arch01.jpg)

**2. Unipreneur Competition, Cambodia in sep, 2024**:

- Our team won Top 6 National Award.

![arch02](/images/arch02.png)

# V. Our Goals
Our goals aims to create mobile application to address the inclusive challenges faced by the deaf and mute community in Cambodia by developing an application that translates between sign language and spoken Khmer. 

![ksl goal](/images/ksl_goal.png)

- We want to create the mobile application for deaf and mute people in community through AI technologies. We're utilizing AI algorithms, image processing, and TTS model, the app will facilitate effective communication, enhancing access to education, healthcare, and employment
opportunities. 
- As what we have done above is a proof of concept and are now focused on refining our algorithms, expanding our sign language database, and building user-friendly mobile applications. 
- This technology has the potential to greatly improve the quality of life and social inclusion for disabled individuals in Cambodia and the region.

---

***Notice***: For full technical details, architecture design, and experimental results, please refer to the complete report:
[[View my Report](/files/KSL_Project.pdf)] and my demo video [[Demo videos](https://drive.google.com/drive/folders/13BYBKfAXmTQ-JzGwgc5eopT5qk_auznO?usp=drive_link)].

---