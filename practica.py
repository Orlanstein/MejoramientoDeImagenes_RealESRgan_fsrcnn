import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Para nombrar bien mis archivos
import os
import re

# Descargar imagen de ejemplo (reemplaza con tu fuente de video)
# url = "https://example.com/low_res_face.jpg"
# response = requests.get(url)
# low_res_img = np.array(Image.open(BytesIO(response.content)))

# Alternativa: Cargar imagen local
low_res_img = cv2.imread("inputImages/nicolas.png")

# Método de super-resolución tradicional (ESRGAN requiere GPU)
def enhance_resolution(img):
    # Upscaling con interpolación
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Mejorar nitidez
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    # Reducción de ruido
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    return denoised

enhanced_frame = enhance_resolution(low_res_img)

# Nombre de la subcarpeta
carpeta = "mejoradas"

# Crear la carpeta si no existe
os.makedirs(carpeta, exist_ok=True)

# Buscar archivos dentro de la subcarpeta
patron = re.compile(r"enhanced_face(\d+)\.jpg")
numeros = []

for archivo in os.listdir(carpeta):
    match = patron.match(archivo)
    if match:
        numeros.append(int(match.group(1)))

nuevo_numero = max(numeros) + 1 if numeros else 1
nuevo_nombre = f"enhanced_face{nuevo_numero}.jpg"
ruta_completa = os.path.join(carpeta, nuevo_nombre)

# Guardar imagen en la subcarpeta
cv2.imwrite(ruta_completa, enhanced_frame)
print(f"Procesamiento completado. Imagen mejorada guardada como {ruta_completa}")