import uvicorn
import keras
import numpy as np
import io
import tensorflow as tf
import base64

from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI(title="HEATMAP API")

@app.get("/")
def homepage():
    html = """
           <html><head><title>HEATMAP API</title></head>
           <body>
           <h1>HEATMAP API</h1>
           <h2>Hello!</h2>
           <h2>Breast Cancer API for heatmap is live</h2>
           </body>
           </html>
           """
    print(os.path.realpath('breast_cancer_classification-sa.h5'))
    return HTMLResponse(content=html, status_code=200)

@app.post("/generate_heatmap")
async def generate_heatmap(image_: UploadFile):
    data = image_.file.read()
    model = keras.models.load_model('breast_cancer_classification-sa.h5')

    image = Image.open(io.BytesIO(data))

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
    grad_model = keras.models.Model(
        [model.inputs],
        [model.get_layer('conv2d_1').output,
         model.get_layer('conv2d_1').output])
    with tf.GradientTape() as tape:
        # Forward pass through grad_model
        last_conv_layer_output, preds = grad_model(image_array)
        
        # Use the predicted class if no index is provided
        # pred_index = tf.argmax(preds[0])
        class_channel = preds[:, predicted_class_index]

        # Compute the gradient of the top predicted class for the input image
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Compute pooled gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel in the feature map array by "how important this channel is"
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap.numpy()
        
        heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
        # Save the image to a BytesIO object
        img_bytes = io.BytesIO()
        heatmap_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return {"heatmap": base64.b64encode(img_bytes.read()).decode('utf-8')}
        
if __name__ == "__main__":
    uvicorn.run("heatmap-api:app",
                host="172.28.168.45",
                port=8000,
                reload=True)