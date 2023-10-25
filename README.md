<img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-00AFF0?logo=Kaggle&logoColor=white&style=flat" /> <img alt="Dash" src="https://img.shields.io/badge/Dash-008DE4?logo=Dash&logoColor=white&style=flat" />  <img alt="Google" src="https://img.shields.io/badge/Google-4285F4?logo=Google&logoColor=white&style=flat" /> <img alt="Plotly" src="https://img.shields.io/badge/Plotly-3F4F75?logo=Plotly&logoColor=white&style=flat" /> 
<img alt="Python" src="https://img.shields.io/badge/Python%20-%2314354C.svg?style=flat-square&logo=python&logoColor=white" /> <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white&style=flat" /> 
 <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=flat" />

<br>

![header](https://user-images.githubusercontent.com/70657426/233868082-147bd7a7-08fd-4744-92ee-b396c42b6771.png)
# Google-Isolated-Sign-Language-Recognition-Kaggle 
Deaf children are often born to hearing parents who do not know sign language. The challenge in this code competition is to help identify signs made in processed videos, which will support the development of mobile apps to help teach parents sign language so they can communicate with their Deaf children.

The competition aims to develop a TensorFlow Lite model for classifying isolated American Sign Language (ASL) signs. The model will be trained on labeled landmark data extracted using the MediaPipe Holistic Solution. The goal is to improve the functionality of the PopSign smartphone game app, which helps parents of deaf children learn ASL and allows anyone interested in learning sign language vocabulary to practice.

By creating a sign language recognizer, participants can enhance the game's interactivity and facilitate learning and communication for players who want to use sign language to connect with their loved ones. Since the app doesn't send user videos to the cloud and performs all inference on the device itself, TensorFlow Lite is the chosen framework to ensure low latency within the game. 


## Project Structure

```bash
google_asl
├── utility_scripts/             
|   ├── feature_engineering.py   
|   └── helper_functions.py   
├── visualization_scripts/  
|   ├── animated_3d_viz.ipynb
|   └── viz_dashboard.ipynb               
├── datasets/  
|   ├── mean_dataset.npy
|       ├── feature_data.npy
|       └── feature_labels.npy
|   ├── spatio_temporal_data/  
|       ├── spatio_temporal_script.ipynb
|       ├── feature_data.npy
|       └── feature_labels.npy
|   └── kaggle_data/      
|       ├── sign_to_prediction_index_map.json
|       ├── train.csv
|       └── train_landmark_files
├── preprocessing_keras_layer/                  
|   └── angular_distance_layers.py            
├── models/
|   ├── gru_asl_model.ipynb                   
|   ├── nn_asl_model.ipynb         
|   └── sequential_model.ipynb             
└── application/
|   ├── mediapipe_landmarks.py
|   └── real_time_asl_recognition.ipynb        
```

## Overview
### Optimal Dataset Generation
The goal of this first step is to create a new dataset that takes into consideration both the temporal and spatial aspect of the ASL training set to optimize the performance of our model.
<p align="center">
<img width="700" alt="Screen Shot 2023-03-27 at 2 02 25 PM" src="https://user-images.githubusercontent.com/70657426/233866270-32581349-d576-4455-9f51-ad5a5c7f16b3.png">
</p>

<br>
 
### Landmark Vizualisation
This part presents an interactive dashboard that allows exploring American Sign Language (ASL) landmarks in 2D space. By selecting different ASL categories and IDs, users can filter the displayed landmarks and study the hand shapes and movements associated with specific signs. The dashboard is built with Plotly and Python, and can be used for educational purposes or to support the development of ASL recognition systems.

https://user-images.githubusercontent.com/70657426/233866585-c9f2dd53-4d5d-4f5f-ad7e-0c5aa7fc7a9b.mov

### Models Building
The goal here is to build a baseline ASL recognition engine using the average frames dataset. We will be leveraging the modularity of the custom keras layer to apply some preprocessing steps to the 3D landarks and for feature extraction. 

<img width="1560" alt="Screen Shot 2023-04-05 at 1 19 31 PM" src="https://user-images.githubusercontent.com/70657426/233866703-7da50050-ac3a-46c6-a95a-c509e361c0a2.png">

### Real-life Application
Use your pre-trained TensorFlow Lite model to perform real-time American Sign Language (ASL) recognition from YOUR webcam video!

<p align="center">
<img width="700" alt="Screen Shot 2023-04-06 at 7 19 50 PM" src="https://user-images.githubusercontent.com/70657426/233866804-289a8734-8c90-4b53-a5e7-d4b067b1842d.png">
</p>

### Built With
- Data Collection & Manipulation: [Pandas](https://pandas.pydata.org)
- Vizualisation: [Plotly](https://plotly.com) & [Dash](https://plotly.com/dash/)
- Feature Engineering: [Keras](https://keras.io)
- Neural Networks: [Tensorflow](https://www.tensorflow.org)
- Deployment: [TFLite](https://www.tensorflow.org/lite)
