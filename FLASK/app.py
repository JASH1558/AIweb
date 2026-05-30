from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import numpy as np
import torch

from PIL import Image

from main import conv

app = Flask(__name__)
CORS(app)

# ======================
# LOAD MODEL
# ======================

model = conv()
model.load_state_dict(
    torch.load(
        "mnist_model.pth",
        map_location=torch.device("cpu")
    )
)
print(next(model.parameters())[0][:5])
model.eval()
print(next(model.parameters())[0][:5])
# ======================
# ROUTE
# ======================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        pixels = np.array(
            data["pixels"],
            dtype=np.uint8
        )
        arr = pixels.reshape(28, 28, 4)

        arr = arr[:, :, :3]

        img = Image.fromarray(arr).convert("L")

        img_array = np.array(img)

        img_array = img_array / 255.0

        image_tensor = torch.tensor(
            img_array,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

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

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500

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
