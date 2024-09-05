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

with st.sidebar:
    st.subheader("Usa un modelo entrenado en Teachable Machine para identificar imágenes")
    
    # Opción para usar la cámara
    cam_ = st.checkbox("Usar Cámara")
    # Opción para subir una imagen desde el archivo
    upload_ = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if cam_:
    img_file_buffer = st.camera_input("Toma una Foto")
else:
    img_file_buffer = None

if img_file_buffer is not None:
    # Lee el buffer de la imagen como una imagen PIL
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # Convierte la imagen PIL a un array numpy
    img_array = np.array(img)

    # Verifica el tipo y la forma del array
    st.write("Forma de img_array:", img_array.shape)
    st.write("Tipo de img_array:", img_array.dtype)

    # Normaliza la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Verifica el tipo y la forma del array normalizado
    st.write("Forma de normalized_image_array:", normalized_image_array.shape)
    st.write("Tipo de normalized_image_array:", normalized_image_array.dtype)

    # Verifica que la forma sea (224, 224, 3)
    if normalized_image_array.shape == (224, 224, 3):
        # Carga la imagen en el array
        data[0] = normalized_image_array
        # Ejecuta la inferencia
        prediction = model.predict(data)
        print(prediction)
        if prediction[0][0] > 0.5:
            st.header('Palma, con Probabilidad: ' + str(prediction[0][0]))
        if prediction[0][1] > 0.5:
            st.header('Ok, con Probabilidad: ' + str(prediction[0][1]))
        if prediction[0][2] > 0.5:
            st.header('JCBG, con Probabilidad: ' + str(prediction[0][2]))
        if prediction[0][3] > 0.5:
            st.header('Vacío, con Probabilidad: ' + str(prediction[0][3]))
    else:
        st.error("La imagen no tiene la forma esperada (224, 224, 3).")
else:
    # Si se sube una imagen
    if upload_ is not None:
        # Lee el archivo subido como una imagen PIL
        img = Image.open(upload_)

        newsize = (224, 224)
        img = img.resize(newsize)
        # Convierte la imagen PIL a un array numpy
        img_array = np.array(img)

        # Verifica el tipo y la forma del array
        st.write("Forma de img_array:", img_array.shape)
        st.write("Tipo de img_array:", img_array.dtype)

        # Normaliza la imagen
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        # Verifica el tipo y la forma del array normalizado
        st.write("Forma de normalized_image_array:", normalized_image_array.shape)
        st.write("Tipo de normalized_image_array:", normalized_image_array.dtype)

        # Verifica que la forma sea (224, 224, 3)
        if normalized_image_array.shape == (224, 224, 3):
            # Carga la imagen en el array
            data[0] = normalized_image_array
            # Ejecuta la inferencia
            prediction = model.predict(data)
            print(prediction)
            if prediction[0][0] > 0.5:
                st.header('Palma, con Probabilidad: ' + str(prediction[0][0]))
            if prediction[0][1] > 0.5:
                st.header('Ok, con Probabilidad: ' + str(prediction[0][1]))
            if prediction[0][2] > 0.5:
                st.header('JCBG, con Probabilidad: ' + str(prediction[0][2]))
            if prediction[0][3] > 0.5:
                st.header('Vacío, con Probabilidad: ' + str(prediction[0][3]))
        else:
            st.error("La imagen no tiene la forma esperada (224, 224, 3).")

