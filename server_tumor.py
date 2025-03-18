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

# ✅ Load TFLite model **only once** at startup
interpreter = None
input_details = None
output_details = None

@app.on_event("startup")
def load_tflite_model():
    global interpreter, input_details, output_details
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

brain_tumor_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Preprocess function
def preprocess_brain_tumor_image(image):
    img = Image.open(image).convert("RGB").resize((299, 299))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# Prediction function
def predict_brain_tumor_tflite(image):
    img_array = preprocess_brain_tumor_image(image)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = brain_tumor_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    gc.collect()  # Free up memory
    return predicted_class, confidence

# FastAPI Endpoint
@app.post("/predict_brain_tumor/")
async def predict_brain_tumor_endpoint(file: UploadFile = File(...)):
    prediction, confidence = predict_brain_tumor_tflite(file.file)
    return {"diagnosis": prediction, "confidence": f"{confidence:.2f}%"}
@app.get("/")
def health_check():
    return {"status": "OK"}
