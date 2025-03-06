import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import shutil
import os

app = FastAPI()

# CORS configuration to allow requests from all sources
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Define preprocessing function
def preprocess_brain_tumor_image(image_path):
    """ Preprocess image for brain tumor model """
    img = Image.open(image_path).resize((299, 299))  # Resize to match training
    img_array = np.asarray(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims
    return img_array

# Define label mappings
brain_tumor_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Prediction function using TFLite
def predict_brain_tumor_tflite(image_path):
    img_array = preprocess_brain_tumor_image(image_path)
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class and confidence
    predicted_class = brain_tumor_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# FastAPI Endpoint
@app.post("/predict_brain_tumor/")
async def predict_brain_tumor_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = predict_brain_tumor_tflite(file_path)
    os.remove(file_path)  # Cleanup

    return {"diagnosis": prediction, "confidence": f"{confidence:.2f}%"}
