import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")

# Mostrar una imagen inicial
image = Image.open('OIG5.jpg')
st.image(image, width=350)

# Barra lateral con solo el texto
with st.sidebar:
    st.subheader("Usa un modelo entrenado en Teachable Machine para identificar imágenes")

# Opción para usar la cámara
cam_ = st.checkbox("Usar Cámara")

# Opción para subir una imagen desde el archivo
upload_ = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

def process_image(img):
    # Redimensiona la imagen
    newsize = (224, 224)
    img = img.resize(newsize)
    # Convierte la imagen PIL a un array numpy
    img_array = np.array(img)
    
    # Verifica si la imagen es en escala de grises o en RGB
    if len(img_array.shape) == 2:
        # Imagen en escala de grises, conviértela a RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Verifica la forma del array
    if img_array.shape != (224, 224, 3):
        st.error(f"La imagen tiene una forma inesperada: {img_array.shape}. Se esperaba (224, 224, 3).")
        return None

    # Normaliza la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    return normalized_image_array

def predict_image(image_array):
    if image_array.shape == (224, 224, 3) and image_array.dtype == np.float32:
        data[0] = image_array
        # Ejecuta la inferencia
        prediction = model.predict(data)
        st.write("Predicción:", prediction)
        return prediction
    else:
        st.error(f"Array tiene una forma o tipo incorrecto: {image_array.shape}, {image_array.dtype}")
        return None

# Contenedores para mostrar resultados
image_container = st.empty()
result_container = st.empty()

# Variables para almacenar la imagen y los resultados
image_display = None
prediction_result = None

if cam_:
    img_file_buffer = st.camera_input("Toma una Foto")
    if img_file_buffer is not None:
        # Lee el buffer de la imagen como una imagen PIL
        img = Image.open(img_file_buffer).convert('RGB')
        normalized_image_array = process_image(img)
        
        if normalized_image_array is not None:
            # Limpia el contenedor de la imagen anterior
            image_container.empty()
            # Muestra la imagen tomada
            image_container.image(img, caption='Foto tomada', use_column_width=True)
            # Realiza la predicción
            prediction_result = predict_image(normalized_image_array)
            if prediction_result is not None:
                # Limpia el contenedor de los resultados anteriores
                result_container.empty()
                if prediction_result[0][0] > 0.5:
                    result_container.header('Palma, con Probabilidad: ' + str(prediction_result[0][0]))
                if prediction_result[0][1] > 0.5:
                    result_container.header('Ok, con Probabilidad: ' + str(prediction_result[0][1]))
                if prediction_result[0][2] > 0.5:
                    result_container.header('JCBG, con Probabilidad: ' + str(prediction_result[0][2]))
                if prediction_result[0][3] > 0.5:
                    result_container.header('Vacío, con Probabilidad: ' + str(prediction_result[0][3]))

if upload_ is not None:
    # Lee el archivo subido como una imagen PIL
    uploaded_image = Image.open(upload_).convert('RGB')
    normalized_image_array = process_image(uploaded_image)
    
    if normalized_image_array is not None:
        # Limpia el contenedor de la imagen anterior
        image_container.empty()
        # Muestra la imagen cargada
        image_container.image(uploaded_image, caption='Imagen cargada.', use_column_width=True)
        # Realiza la predicción
        prediction_result = predict_image(normalized_image_array)
        if prediction_result is not None:
            # Limpia el contenedor de los resultados anteriores
            result_container.empty()
            if prediction_result[0][0] > 0.5:
                result_container.header('Palma, con Probabilidad: ' + str(prediction_result[0][0]))
            if prediction_result[0][1] > 0.5:
                result_container.header('Ok, con Probabilidad: ' + str(prediction_result[0][1]))
            if prediction_result[0][2] > 0.5:
                result_container.header('JCBG, con Probabilidad: ' + str(prediction_result[0][2]))
            if prediction_result[0][3] > 0.5:
                result_container.header('Vacío, con Probabilidad: ' + str(prediction_result[0][3]))
