from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import io
from collections import Counter
import keras
keras.config.enable_unsafe_deserialization()

# Load models
CNN_1 = load_model(r"D:\Faculty\graduation project\Disease-Prediction-Model-master-main\CNN.keras")
CNN_2 = load_model(r"D:\Faculty\graduation project\Disease-Prediction-Model-master-main\CNN2.keras")
vgg_model = load_model(r"D:\Faculty\graduation project\Disease-Prediction-Model-master-main\VGG16.keras")

models = [CNN_1, CNN_2, vgg_model]

app = Flask(__name__)

def preprocess_image(image_bytes, target_size=(64, 64)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  
    return image_array

def predict_single_image(model, image):
    image_batch = np.expand_dims(image, axis=0)
    prediction = (model.predict(image_batch) > 0.5).astype('int32')[0][0]
    return 'Parasitized' if prediction == 0 else 'Uninfected'

def majority_voting(predictions):
    vote_counts = Counter(predictions)
    most_common = vote_counts.most_common(1)[0][0]
    return most_common

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    image = preprocess_image(image_bytes)  # output shape (64, 64, 3)

    
    predictions = [predict_single_image(m, image) for m in models]
    
    
    final_result = majority_voting(predictions)

    return jsonify({"diagnosis": final_result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
