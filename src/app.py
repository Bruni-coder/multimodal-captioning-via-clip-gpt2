from flask import Flask, render_template, request
from predictor import MultiModalPredictor
from PIL import Image
import os

app = Flask(__name__)
predictor = MultiModalPredictor(model_path="../gpt2_epoch8")

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        file = request.files["image"]
        prompt = request.form["prompt"]
        image = Image.open(file.stream).convert("RGB")
        result = predictor.generate_text_from_image(image, prompt)
    return render_template("../templates/index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
