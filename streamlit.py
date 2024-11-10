import streamlit as st
import keras
from PIL import Image
import numpy as np
import io

def main():
    model = keras.models.load_model("models/breast_cancer_classification-sa.h5")

    st.title("Breast Cancer Classification")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(data)).resize((28, 28))
        st.image(img)

        img = load_img(img_path, target_size=(224, 224))

        x = img_to_array(img)
        x = x / 255.0  
        x = np.expand_dims(x, axis=0)  

        preds = model.predict(x)
        predicted_class_index = np.argmax(preds[0])

        class_labels = classes_  
        predicted_class_label = class_labels[predicted_class_index]

        print("Predicted class:", predicted_class_label)

        arr = np.array(img)
        
        # arr = np.average(arr, axis=-1)
        arr = arr.reshape(1, 28, 28, 1)
        prediction = model.predict(arr)
        prediction = np.argmax(prediction)
        st.subheader(prediction)
        
if __name__ == "__main__":
    main()