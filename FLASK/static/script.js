const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", {
    willReadFrequently: true
});
const predictionText = document.getElementById("prediction");
const clearBtn = document.getElementById("clearBtn");

// Canvas resolution
canvas.width = 28;
canvas.height = 28;

// Drawing state
let drawing = false;
let timeout;

// Background
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Brush settings
ctx.strokeStyle = "white";
ctx.lineWidth = 4;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.lineJoin = "round";
ctx.lineCap = "round";
ctx.imageSmoothingEnabled = false;

// =========================
// START DRAWING
// =========================
function getMousePos(e) {

    const rect = canvas.getBoundingClientRect();

    return {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height)
    };
}

canvas.addEventListener("mousedown", startDrawing);

function startDrawing(e) {

    drawing = true;

    const pos = getMousePos(e);

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}


// =========================
// STOP DRAWING
// =========================

canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

function stopDrawing() {
    drawing = false;
}

// =========================
// DRAW
// =========================

canvas.addEventListener("mousemove", draw);

function draw(e) {

    if (!drawing) return;

    const pos = getMousePos(e);

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    triggerPrediction();
}

// =========================
// CLEAR BUTTON
// =========================

clearBtn.addEventListener("click", clearCanvas);

function clearCanvas() {

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // reset prediction text
    document.getElementById("mainPrediction").innerText = "-";

    // reset probability bars
    for (let i = 0; i < 10; i++) {

        document.getElementById(`fill${i}`).style.width = "0%";
        document.getElementById(`text${i}`).innerText = "0%";
    }

    // optional: stop pending prediction calls
    clearTimeout(timeout);
}

// =========================
// DEBOUNCE PREDICTION
// =========================

function triggerPrediction() {

    clearTimeout(timeout);

    timeout = setTimeout(() => {

        sendToNetwork();

    }, 150);
}

// =========================
// SEND TO FLASK BACKEND
// =========================

function sendToNetwork() {

    // Extract pixel data
    const imageData = ctx.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
    );

    // Convert Uint8ClampedArray → normal array
    const pixels = Array.from(imageData.data);

    // Send to backend
    fetch("/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            pixels: pixels
        })

    })

    .then(response => response.json())
.then(data => {

    const probs = data.probabilities;

    document.getElementById("mainPrediction")
    .innerText = data.digit;

    for (let i = 0; i < 10; i++) {

        const percent = Math.round(probs[i] * 100);

        document.getElementById(`fill${i}`)
        .style.width = percent + "%";

        document.getElementById(`text${i}`)
        .innerText = percent + "%";
    }

})

    .catch(error => {

        console.error("Error:", error);

    });
}
