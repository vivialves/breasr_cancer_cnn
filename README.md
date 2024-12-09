# Breast Cancer Classification Using Convolutional Neural Networks (CNNs)

This project focuses on classifying breast cancer mammography images into multiple classes using Convolutional Neural Networks (CNNs). By leveraging state-of-the-art deep learning architectures and data augmentation techniques, this study aims to provide insights into the diagnostic process of breast cancer while emphasizing model interpretability. This is a first version of this study and it is open to improvements.

# Table of Contents
Introduction
Features
Dataset
Model Architectures
Techniques Used
Results
Installation
Usage
Future Work
Contributing
License
Example of heatmap
Link to access

## Introduction
Breast cancer is a critical global health challenge. Early and accurate detection is key to improving survival rates. This project uses deep learning techniques, particularly CNNs, to classify mammography images across 8 distinct classes. The models evaluated include ShuffleNet, DenseNet, EfficientNet, VGG16, AlexNet, and a custom-designed architecture. 

## Features

 - Multi-Class Classification: Identifies breast cancer classes based on mammographic density and malignancy.
 - Explainability: Uses Grad-CAM for model interpretability, highlighting regions of focus.
 - Data Augmentation: Enhances model performance on small datasets.
 - Metrics Evaluation: Performance assessed via accuracy, precision, recall, and F1-score.

## Dataset
 - Source: https://data.mendeley.com/datasets/x7bvzv6cvr/1
 - Paper: https://www.sciencedirect.com/science/article/pii/S2352340920308222

 - Structure: The dataset includes images divided into the following classes:

1. Density1Benign
2. Density1Malign
3. Density2Benign
4. Density2Malign
5. Density3Benign
6. Density3Malign
7. Density4Benign
8. Density4Malign

Preprocessing: Images resized to 224x224 and 227X227 pixels for compatibility with CNN architectures.

## Model Architectures
The following architectures were trained and evaluated:

 - Custom Simple Architecture: Designed from scratch for this study.
 - ShuffleNet
 - DenseNet
 - EfficientNet
 - VGG16
 - AlexNet

## Techniques Used
 - Hyperparameter Tuning: Batch size, learning rate, and epochs were optimized.
 - Data Augmentation: Techniques included rotation, flipping, and zooming.
 - Transfer Learning: Applied to VGG16 and EfficientNet.
 - Grad-CAM: Provided heatmaps for explainability and decision validation.
   
## Results
 - Top-Performing Models: Simple architecture, ShuffleNet, and DenseNet achieved the best results.
   
## Key Metrics:
 - Accuracy: Up to 98%.
 - Precision, Recall, and F1-Score: Demonstrated strong classification ability.
 - Grad-CAM Analysis: Confirmed the reliability of model predictions and identified potential biases.

## Main libraries

 - Keras
 - Tensorflow
 - FastAPI
 - Streamlit

* API was deployed in App Engine from Google Cloud Platform

## Installation
 - Clone the Repository:
git clone https://github.com/vivialves/breast_cancer_cnn
cd breast-cancer-cnn

 - Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


 - Install Dependencies:
pip install -r requirements.txt


## Usage

 - Train the Model:
Run the jupyter notebooks in notebooks folder

 - Generate Heatmaps:

cd app
fastapi run api.py

 - Streamlit

cd app
streamlit run streamlit.py   


## Future Work
 - Expand the dataset for improved generalizability.
 - Integrate additional imaging modalities like ultrasound or MRI.
 - Develop further explainability tools for medical practitioners.
 - Soon I will start to work in version 2.
   
## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License
This project is licensed under the CC BY 4.0 licence.

## Example of heatmap json:

{
  "heatmap": "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"
}

## Link to access: 
https://classbreastcancer.streamlit.app/
