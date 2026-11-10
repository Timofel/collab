from pathlib import Path
import base64
import io

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = Flask(__name__)

# Пути к моделям
SIMPLE_MODEL_PATH = Path(
    r"C:\Users\user\Desktop\detects_ns\simple_version\nut_defect_model.keras"
)

COMPLEX_MODEL_PATH = Path(
    r"C:\Users\user\Desktop\detects_ns\no_simple_version"
    r"\runs\detect\defect_detector\weights\best.pt"
)

IMG_SIZE = (180, 180)

# Укажите названия классов модели .keras в правильном порядке.
# Они должны совпадать с папками, на которых обучалась модель.
SIMPLE_CLASS_NAMES = ["класс_0", "класс_1"]

# Загрузка моделей при запуске сервера
simple_model = None
complex_model = None

if SIMPLE_MODEL_PATH.exists():
    simple_model = tf.keras.models.load_model(SIMPLE_MODEL_PATH)
else:
    print(f"Предупреждение: не найдена простая модель: {SIMPLE_MODEL_PATH}")

if COMPLEX_MODEL_PATH.exists():
    complex_model = YOLO(str(COMPLEX_MODEL_PATH))
else:
    print(f"Предупреждение: не найдена сложная модель: {COMPLEX_MODEL_PATH}")


def predict_simple(image):
    """Бинарная классификация моделью Keras."""
    if simple_model is None:
        raise FileNotFoundError(f"Не найдена модель: {SIMPLE_MODEL_PATH}")

    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)[None, ...]

    # Rescaling(1/255) уже включён в Keras-модель.
    probability = float(simple_model.predict(array, verbose=0)[0][0])

    class_index = 1 if probability >= 0.5 else 0
    confidence = probability if class_index == 1 else 1 - probability

    label = (
        SIMPLE_CLASS_NAMES[class_index]
        if class_index < len(SIMPLE_CLASS_NAMES)
        else f"Класс {class_index}"
    )

    return label, confidence * 100


def predict_complex(image):
    """YOLO-предсказание и изображение с обозначениями."""
    if complex_model is None:
        raise FileNotFoundError(f"Не найдена модель: {COMPLEX_MODEL_PATH}")

    predictions = complex_model.predict(
        source=np.asarray(image),
        conf=0.25,
        save=False,
        verbose=False,
    )

    prediction = predictions[0]

    if prediction.boxes is not None and len(prediction.boxes) > 0:
        confidences = prediction.boxes.conf.cpu().numpy()
        class_ids = prediction.boxes.cls.cpu().numpy().astype(int)

        best_index = int(np.argmax(confidences))
        class_id = class_ids[best_index]
        confidence = float(confidences[best_index])
        label = prediction.names[class_id]
    else:
        label = "Объекты не обнаружены"
        confidence = 0.0

    # YOLO наносит рамки, названия классов и confidence на изображение.
    annotated = prediction.plot()
    annotated_image = Image.fromarray(annotated[:, :, ::-1])

    return label, confidence * 100, annotated_image


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None
    error = None
    selected_model = request.form.get("model", "simple")

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Выберите изображение."
        else:
            try:
                image = Image.open(uploaded_file.stream).convert("RGB")

                if selected_model == "complex":
                    label, confidence, display_image = predict_complex(image)
                    model_title = "Сложная модель (YOLO)"
                else:
                    label, confidence = predict_simple(image)
                    display_image = image
                    model_title = "Простая модель (Keras)"

                result = {
                    "label": label,
                    "confidence": round(confidence, 2),
                    "model": model_title,
                }

                preview = io.BytesIO()
                display_image.thumbnail((1000, 1000))
                display_image.save(preview, format="JPEG")
                image_data = base64.b64encode(preview.getvalue()).decode("utf-8")

            except (UnidentifiedImageError, OSError):
                error = "Не удалось прочитать изображение. Попробуйте JPG или PNG."
            except Exception as exc:
                error = f"Ошибка обработки: {exc}"

    return render_template(
        "main.html",
        result=result,
        image_data=image_data,
        error=error,
        selected_model=selected_model,
    )


if __name__ == "__main__":
    print("Сервер запущен: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)