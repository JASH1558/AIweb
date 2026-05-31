from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import numpy as np
import torch

from PIL import Image

from main import conv

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False

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

    print("PREDICT CALLED", flush=True)

    try:

        data = request.get_json()
        print("JSON received", flush=True)

        pixels = np.array(data["pixels"], dtype=np.uint8)
        print("Pixels shape:", pixels.shape, flush=True)

        arr = pixels.reshape(28, 28, 4)

        arr = arr[:, :, :3]

        img = Image.fromarray(arr).convert("L")

        img_array = np.array(img)

        img_array = img_array / 255.0

        image_tensor = torch.tensor(
            img_array,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        print("Tensor shape:", image_tensor.shape, flush=True)

        with torch.no_grad():

            print("Running model...", flush=True)

            output = model(image_tensor)

            print("Model finished", flush=True)

            probabilities = torch.softmax(output, dim=1)

            prediction = torch.argmax(
                probabilities,
                dim=1
            )

        return jsonify({
            "digit": int(prediction.item()),
            "probabilities": probabilities.squeeze().tolist()
        })

    except Exception as e:

        print("ERROR:", repr(e), flush=True)

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
