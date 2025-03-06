import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gc

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model once (stays in memory)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Define label mappings
brain_tumor_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Define preprocessing function
def preprocess_brain_tumor_image(image):
    """ Preprocess image for brain tumor model """
    img = Image.open(image).convert("RGB").resize((299, 299))  # Convert to RGB & resize
    img_array = np.asarray(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    return img_array.astype(np.float32)

# Prediction function using TFLite
def predict_brain_tumor_tflite(image):
    """ Run inference using TensorFlow Lite """
    img_array = preprocess_brain_tumor_image(image)

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Determine class and confidence
    predicted_class = brain_tumor_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    gc.collect()  # Free up memory

    return predicted_class, confidence

# FastAPI Endpoint
@app.post("/predict_brain_tumor/")
async def predict_brain_tumor_endpoint(file: UploadFile = File(...)):
    """ Endpoint to receive an image, process it, and return the prediction. """
    prediction, confidence = predict_brain_tumor_tflite(file.file)
    return {"diagnosis": prediction, "confidence": f"{confidence:.2f}%"}
