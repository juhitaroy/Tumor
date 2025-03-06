import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI()

# Load Brain Tumor model
brain_tumor_model = tf.keras.models.load_model("brain_tumor_model.h5")

# Define preprocessing function
def preprocess_brain_tumor_image(image_path):
    """ Preprocess image for brain tumor model """
    img = Image.open(image_path).resize((299, 299))  # Resize to match training
    img_array = np.asarray(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims
    return img_array

# Define label mappings
brain_tumor_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Prediction function
def predict_brain_tumor(image_path):
    img_array = preprocess_brain_tumor_image(image_path)
    predictions = brain_tumor_model.predict(img_array)
    predicted_class = brain_tumor_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# FastAPI Endpoint
@app.post("/predict_brain_tumor/")
async def predict_brain_tumor_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = predict_brain_tumor(file_path)
    os.remove(file_path)  # Cleanup

    return {"diagnosis": prediction, "confidence": f"{confidence:.2f}%"}
