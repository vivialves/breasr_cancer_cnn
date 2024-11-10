import streamlit as st
import keras
from PIL import Image
import numpy as np
import io
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array



def main():
    model = keras.models.load_model("models/breast_cancer_classification-sa.h5")

    st.title("Breast Cancer Classification")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(data))
        small_image = image.resize((112, 112))
        st.image(small_image)
        if image.mode != "RGB":
            image = image.convert("RGB")     
        image_array = img_to_array(image)
        # if image_array.ndim == 2:
            # image_array = np.stack((image_array,) * 3, axis=-1)
        image_array = tf.image.resize(image_array, size=[224, 224])
        # if image_array.dtype != 'float32':
            # image_array = image_array.astype('float32')
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
        
if __name__ == "__main__":
    main()