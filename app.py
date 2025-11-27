from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Initialize the Flask application with the current module name
app = Flask(__name__)

# Restore the trained keras model from disk
model = load_model("emotion_model.h5")

# Emotion classification model(must reflect model output)
EMOTIONS = ["Angry", "Disgust", "Afraid", "Happy", "Sad", "Surprise", "Neutral"]

# Follow-up messages for each emotion
DESCRIPTIONS = {
    "Happy": "You look joyful today! best feeling ever.",
    "Sad": "it seems you're feeling down and that's normal, be gentle with yourself",
    "Angry": "I can tell you're upset, try to pause and calm yourself.",
    "Surprise": "You look amazed!",
    "Neutral": "You are in a balanced state of mind. That is awesome!.",
    "Afraid": "Do not be worried, things will fall in place again, when you dont expect it.",
    "Disgust": "Sometimes, things dont just sit right with us."
}

# To create an optional Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """
    Converts image into a format suitable for MobileNetV2:
    - Load image
    - Resize to 224x224
    - Normalize pixel values
    - Expand dimensions for model input
    """
    image = Image.open(image_path).convert("L")
    image = image.resize((48, 48))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/")
def index():
    """Load the main homepage (index.html)."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and returns the predicted emotion."""
    

    file = request.files["image"]


    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = preprocess_image(filepath)

    # Make model prediction
    predictions = model.predict(img)

    # Get emotion with highest probability
    emotion_idx = np.argmax(predictions)
    emotion = EMOTIONS[emotion_idx]

    # Get description text
    description = DESCRIPTIONS.get(emotion, "")

    # To Return a JSON file
    return jsonify({
        "emotion": emotion,
        "description": description
    })

# flask server
# commented out incase I consider vercel deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
