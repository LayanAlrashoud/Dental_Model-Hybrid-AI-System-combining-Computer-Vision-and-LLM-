import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER,
    OUTPUT_ORIGINAL_FOLDER,
    OUTPUT_PREDICTED_FOLDER,
    MODEL_PATH,
    ALLOWED_EXTENSIONS,
)

from services.yolo_service import run_segmentation
from services.report_service import generate_dental_report

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

for folder in [UPLOAD_FOLDER, OUTPUT_ORIGINAL_FOLDER, OUTPUT_PREDICTED_FOLDER]:
    os.makedirs(folder, exist_ok=True)


def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None:
            error_message = "No file was uploaded."
            return render_template("index.html", error=error_message)

        if uploaded_file.filename == "":
            error_message = "Please choose an image first."
            return render_template("index.html", error=error_message)

        if not allowed_file(uploaded_file.filename):
            error_message = "Unsupported file type. Please upload a valid image."
            return render_template("index.html", error=error_message)

        try:
            filename = secure_filename(uploaded_file.filename)
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(upload_path)

            result_data = run_segmentation(
                model_path=MODEL_PATH,
                image_path=upload_path,
                output_original_folder=OUTPUT_ORIGINAL_FOLDER,
                output_predicted_folder=OUTPUT_PREDICTED_FOLDER,
            )

            report_data = generate_dental_report(result_data)

            return render_template(
                "result.html",
                original_image=f"outputs/original/{result_data['original_filename']}",
                predicted_all_image=f"outputs/predicted/{result_data['predicted_all_filename']}",
                detections=result_data.get("detections", []),
                single_detection_images=result_data.get("single_detection_images", []),
                findings_summary=report_data.get("findings_summary", {}),
                report=report_data,
            )

        except Exception as e:
            error_message = f"An error occurred while processing the image: {str(e)}"
            return render_template("index.html", error=error_message)

    return render_template("index.html", error=error_message)


if __name__ == "__main__":
    app.run(debug=True)