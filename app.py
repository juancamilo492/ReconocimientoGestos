import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")
#st.write("Versión de Python:", platform.python_version())
image = Image.open('OIG5.jpg')
st.image(image, width=350)
with st.sidebar:
    st.subheader("Usando un modelo entrenado en teachable Machine puedes Usarlo en esta app para identificar")
    
cam_ = st.checkbox("Usar Cámara")
if cam_ :
   img_file_buffer = st.camera_input("Toma una Foto")
else :
   img_file_buffer = None

uploaded_image = st.file_uploader("Cargar Imagen:", type=["png", "jpg"])

if uploaded_image is not None:

    img1 = st.image(uploaded_image, caption='Imagen cargada.', use_column_width=True)
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    img = Image.open(img1)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('Palma, con Probabilidad: '+str( prediction[0][0]) )
    if prediction[0][1]>0.5:
      st.header('Ok, con Probabilidad: '+str( prediction[0][1]))
    if prediction[0][2]>0.5:
      st.header('JCBG, con Probabilidad: '+str( prediction[0][2]))
    if prediction[0][3]>0.5:
      st.header('Vacío, con Probabilidad: '+str( prediction[0][3]))
    
    



if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('Palma, con Probabilidad: '+str( prediction[0][0]) )
    if prediction[0][1]>0.5:
      st.header('Ok, con Probabilidad: '+str( prediction[0][1]))
    if prediction[0][2]>0.5:
      st.header('JCBG, con Probabilidad: '+str( prediction[0][2]))
    if prediction[0][3]>0.5:
      st.header('Vacío, con Probabilidad: '+str( prediction[0][3]))


