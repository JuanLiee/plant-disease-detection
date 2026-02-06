from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from model import predict_image, DISEASE_TREATMENTS
from ollama_ai import ask_ollama

# ================= CONFIG =================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

CONFIDENCE_THRESHOLD = 0.15  
MAX_RESULTS = 3             

# ================= HELPERS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = []
    explanation = ""
    image_url = ""
    treatment = {"chemical": [], "organic": [], "prevention": []}
    warning = ""
    error = ""

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image uploaded"
            return render_template("index.html", error=error)

        file = request.files["image"]
        if file.filename == "":
            error = "No image selected"
            return render_template("index.html", error=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Only png/jpg/jpeg allowed."
            return render_template("index.html", error=error)

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        image_url = path  

        # ===== AI PREDICTION =====
        try:
            predictions = predict_image(path)
        except Exception as e:
            print(e)
            error = "Model failed to analyze image"
            return render_template("index.html", error=error)

        # Format results
        result = [(label, round(conf * 100, 2)) for label, conf in predictions[:MAX_RESULTS]]

        # ===== TOP PREDICTION =====
        top_label, top_conf = predictions[0]

        # ===== OLLAMA EXPLANATION =====
        try:
            explanation = ask_ollama(top_label)
        except Exception as e:
            print(e)
            explanation = "Failed to get explanation from Ollama."

        # ===== LOOKUP OBAT & SOLUSI =====
        treatment = DISEASE_TREATMENTS.get(top_label, {"chemical": [], "organic": [], "prevention": []})

        # ===== WARNING =====
        if top_conf < CONFIDENCE_THRESHOLD:
            warning = "⚠️ Prediction confidence is low. Results may be inaccurate due to image quality."

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        treatment=treatment,
        image_url=image_url,
        warning=warning,
        error=error
    )

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
