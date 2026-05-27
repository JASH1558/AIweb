from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import numpy as np
import torch

from PIL import Image

from main import NeuralNetwork

app = Flask(__name__)
CORS(app)

# ======================
# LOAD MODEL
# ======================

model = NeuralNetwork()

model.load_state_dict(
    torch.load("mnist_model.pth")
)

model.eval()

# ======================
# ROUTE
# ======================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    pixels = np.array(
        data["pixels"],
        dtype=np.uint8
    )

    # RGBA canvas
    arr = pixels.reshape(400, 400, 4)

    # remove alpha channel
    arr = arr[:, :, :3]

    # grayscale
    img = Image.fromarray(arr).convert("L")

    img_array = np.array(img)

    # crop digit
    coords = np.argwhere(img_array > 30)

    if coords.size > 0:

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        img_array = img_array[
            y_min:y_max+1,
            x_min:x_max+1
        ]

    # padding
    padded = np.pad(
        img_array,
        ((20, 20), (20, 20)),
        mode="constant"
    )

    # resize
    img = Image.fromarray(padded).resize((28, 28))

    img_array = np.array(img)

    # normalize
    img_array = img_array / 255.0

    # shape:
    # (1, 1, 28, 28)

    image_tensor = torch.tensor(
        img_array,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    # ======================
    # PREDICTION
    # ======================

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)

        prediction = torch.argmax(
            probabilities,
            dim=1
        )

    return jsonify({

        "digit": int(prediction.item()),

        "probabilities":
        probabilities.squeeze().tolist()

    })

# ======================
# HOME
# ======================

@app.route("/")
def home():

    return render_template("index.html")

# ======================
# RUN
# ======================

if __name__ == "__main__":

    app.run(debug=True)
