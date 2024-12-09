import streamlit as st
import keras
import requests
import numpy as np
import io
import tensorflow as tf
import base64
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


def generate_heatmap_streamlit(image_file):
    """
    :input: image from web page uploaded
    :return: It returns the heatmap image in PIL format.
    """
    files = {'image_': image_file}
    # endpoint local host - used wsl
    #response = requests.post("http://128.0.0.0:8000/generate_heatmap", files=files)
    # endpoint google cloud
    response = requests.post("https://api-442605-p4.nn.r.appspot.com/generate_heatmap", files=files)
    heatmap_data = response.json()["heatmap"]

    # Decode the base64-encoded image
    decoded_img = base64.b64decode(heatmap_data)
    heatmap_img = Image.open(io.BytesIO(decoded_img))
    return heatmap_img

def main() -> None:
    """
    Function called by streamlit.
    """
    path = os.path.realpath('breast_cancer_classification-sa.h5')
    model = keras.models.load_model(path)

    st.title("Machine Learning Classification")
    st.markdown('''
                :green[Breast] :red[Cancer] :orange[Classification] :green[With] :blue[a] :violet[Simple]  :orange[Architecture]
                ''')
    st.markdown(" :female-student::arrow_right::female-doctor:")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(data))
        small_image = image.resize((112, 112))
        st.image(small_image)
        if image.mode != "RGB":
            image = image.convert("RGB")     
        image_array = img_to_array(image)
        image_array = tf.image.resize(image_array, size=[224, 224])
        image_array = image_array / 255.0  
        image_array = np.expand_dims(image_array, axis=0)  
        preds = model.predict(image_array)
        predicted_class_index = np.argmax(preds[0])
        class_labels = ['Density 1 Benign',
                        'Density 1 Malignant',
                        'Density 2 Benign',
                        'Density 2 Malignant',
                        'Density 3 Benign',
                        'Density 3 Malignant',
                        'Density 4 Benign',
                        'Density 4 Malignant']
        predicted_class_label = class_labels[predicted_class_index]
        print("Predicted class:", predicted_class_label)
        st.subheader("Predicted class:")
        st.text(predicted_class_label)
        st.divider()
        st.subheader("HeatMap")
        heatmap = generate_heatmap_streamlit(data)
        fig, ax = plt.subplots()
        ax.matshow(heatmap)
        st.pyplot(fig)
        st.divider()
        st.write('**Disclaimer:** This tool is for educational purposes only and should not be used as a substitute for professional medical             advice. Please consult with a healthcare provider for any health concerns.')
        
if __name__ == "__main__":
    main()