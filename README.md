<img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-00AFF0?logo=Kaggle&logoColor=white&style=flat" /> <img alt="Dash" src="https://img.shields.io/badge/Dash-008DE4?logo=Dash&logoColor=white&style=flat" />  <img alt="Google" src="https://img.shields.io/badge/Google-4285F4?logo=Google&logoColor=white&style=flat" /> <img alt="Plotly" src="https://img.shields.io/badge/Plotly-3F4F75?logo=Plotly&logoColor=white&style=flat" /> 
<img alt="Python" src="https://img.shields.io/badge/Python%20-%2314354C.svg?style=flat-square&logo=python&logoColor=white" /> <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white&style=flat" /> 
 <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=flat" />

# Google-Isolated-Sign-Language-Recognition-Kaggle
The goal of this competition is to classify isolated American Sign Language (ASL) signs. We create a TensorFlow Lite model trained on labeled landmark data extracted using the MediaPipe Holistic Solution to improve the ability of PopSign to help relatives of deaf children learn basic signs and communicate better with their loved ones.

## Project Structure

```bash
google_asl
├── utility_scripts/             
|   ├── feature_engineering.py   
|   └── helper_functions.py   
├── visualization_scripts/  
|   ├── animated_3d_viz.py
|   └── viz_dashbiard.py               
├── datasets/  
|   ├── mean_dataset.npy              
|   ├── spatio_temporal_script.py   
|   ├── spatio_temporal_data.py   
|   └── kaggle_data/      
|       ├── sign_to_prediction_index_map.json
|       ├── train.csv
|       └── train_landmark_files
├── preprocessing_keras_layer/                  
|   └── angular_distance_layers.py            
├── models/
|   ├── gru_asl_model.py                   
|   ├── nn_asl_model.py         
|   └── sequential_model.py             
└── application/
|   └── real_time_asl_recognition.py        
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
