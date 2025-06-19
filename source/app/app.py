from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import joblib
import cv2
import numpy as np
import benford_no_quant_v2 as nq2

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta para servir imágenes del directorio de uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_url = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No se ha subido ninguna imagen.")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="Nombre de archivo vacío.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = detectar_ia(filepath)
        image_url = f'/uploads/{filename}'  # Ruta correcta para mostrar la imagen

    return render_template('index.html', result=result, image_url=image_url)

# Carga el modelo previamente entrenado
model = joblib.load('all_nq2.pkl')

# Función de detección IA
def detectar_ia(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    feature_vector = nq2.get_feature_vector(image, [10, 20, 40, 60])

    # Asegura que sea 2D para scikit-learn
    flattened = feature_vector.cpu().numpy().flatten().reshape(1, -1)

    proba = model.predict_proba(flattened)[0]
    prediction = model.predict(flattened)[0]

    print("Predicción:", prediction)
    print("Probabilidad:", proba)

    label = "Imagen generada por IA" if prediction == 1 else "Imagen real"
    return f"{label} ({round(proba[1] * 100, 2)}% IA)"

if __name__ == "__main__":
    from threading import Thread
    import webbrowser
    import time

    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:5001")

    Thread(target=open_browser).start()
    app.run(host="0.0.0.0", port=5001)
