from flask import Flask, render_template, request
from models.vit import load_vit_model

app = Flask(__name__)
extractor, model = load_vit_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        # Process video and make predictions (dummy implementation)
        prediction = "Action: Running"
        return render_template("index.html", prediction=prediction)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)