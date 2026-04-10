import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

STATIC_FOLDER = os.path.join(BASE_DIR, "static")

UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")
OUTPUT_ORIGINAL_FOLDER = os.path.join(STATIC_FOLDER, "outputs", "original")
OUTPUT_PREDICTED_FOLDER = os.path.join(STATIC_FOLDER, "outputs", "predicted")

MODEL_PATH = os.path.join(BASE_DIR, "model", "dental_segmentation_model.pt")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}