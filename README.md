# Mejoramiento de imagenes con RealESRgan_x4plus y fsrcnn_x2

El repositorio usa dos modelos de mejoramiento de imagenes:

- fsrcnn_x2.pb
- RealESRGAN_x4plus.pth


### Screenshoots

Imagen Original:

![Imagen original](/screenshoots/nicolas.png)

Imagen mejorada con RealESRgan:

![Imagen mejorada](/screenshoots/nicolasmejorado.png)

Imagen mejorada con fsrcnn_x2:

![Imagen mejorada](/screenshoots/enhanced_face1.jpg)

## fsrcnn_x2.pb

FSRCNN es una mejora respecto al modelo anterior SRCNN (Red Neuronal Convolucional de Superresolución), que se centra en aumentar la velocidad de procesamiento sin sacrificar la calidad de la imagen. FSRCNN introduce una nueva arquitectura que incluye una capa de contracción inicial que reduce la dimensionalidad de la entrada, seguida de una serie de capas convolucionales y, finalmente, una capa de expansión que restaura las dimensiones para producir una salida de alta resolución. Este diseño permite a FSRCNN alcanzar un rendimiento en tiempo real sin sacrificar la calidad de los resultados. Es especialmente adecuado para aplicaciones donde la velocidad y la calidad de la imagen son importantes, como en la mejora de vídeo y los videojuegos. [https://www.linkedin.com/pulse/analyzing-image-upscaling-algorithms-comprehensive-study-samuel-sgmvf/]

## RealESRGAN_x4plus.pth

Real‑ESRGAN es un modelo de aprendizaje automático que amplía una imagen con una pérdida mínima de calidad. [https://aihub.qualcomm.com/models/real_esrgan_general_x4v3]

## Instalación

Para la obtencion del modelo se ha proporcionado el modelo ```RealESRGAN_x4plus.pth``` en el repositorio. 
Adicionalmente, se puede descargar de los siguientes enlaces

- https://aihub.qualcomm.com/models/real_esrgan_general_x4v3
- https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.3.0

### Python

Se utiliza la version 3.10 de python


### Entorno virtual

Se debe crear un entorno virtual para la instalacion de las dependencias de python en la version 3.10.

#### Windows:
 - Creacion del venv, cambiar por su ruta equivalente:
 ```
 "C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe" -m venv realesrgan-venv
 ```
 - Activacion del entorno virtual:
 ```
 .\realesrgan-venv\Scripts\Activate.ps1
 ```
#### Linux:
 - Verificacion de disponibilidad de la version de python:
 ```
 python3.10 --version
 ```
 Si no esta instalada, se debe instalar:
 ```
 sudo apt update
 sudo apt install python3.10 python3.10-venv
 ```
 - Creacion del venv, cambiar por su ruta equivalente:
 ```
 python3.10 -m venv realesrgan-venv
 ```
 - Activacion del entorno virtual:
 ```
 source realesrgan-venv/bin/activate
 ```

## Instalacion de dependencias

### Actualizacion de pip:
```
pip install --upgrade pip
```

### Instalacion de dependencias:
```
pip install -r requirements.txt
```


## Arreglar librerias

Las librerias **py-real-esrgan** y **

### Arreglar py-real-esrgan

1. Ir a la carpeta del entorno virtual ```.venv\lib\site-packages\py_real_esrgan``` y abrir el archivo ```model.py```
2. En la linea 6, o donde dice ```from huggingface_hub import hf_hub_url, cached_download```
eliminar ```, cached_download```
3. Buscar todo lo que diga ```cached_download``` y reemplazar por ```hf_hub_url```

### Arreglar basicsr

1. Ir a la carpeta del entorno virtual
```
.venv\lib\site-packages\basicsr\data\
```
2. Abrir degradations.py
3. En la linea 8, o donde dice ```from torchvision.transforms.functional_tensor import rgb_to_grayscale``` cambiar por ```from torchvision._transforms.functional_tensor import rgb_to_grayscale```


## Ejecutar scripts
En este punto se asume que ya se hicieron todos los pasos anteriores, desde la creacion del entorno virtual, la activacion del mismo, la instalacion de dependencias y el arreglo de algunas librerias

### practica.py
Para ejecutar este script, se debe de abrir la terminal del proyecto y ejecutar el siguiente comando:

```
python practica.py
```

### Prueba_realesrgan.py
Para ejecutar este script, se debe de abrir la terminal del proyecto y ejecutar el siguiente comando:
```
python Prueba_realesrgan.py
```

### Prueba_realesrgan2.py
Para ejecutar este script, se debe de abrir la terminal del proyecto y ejecutar el siguiente comando **con los siguiente parametros**:

``` 
python Prueba_realesrgan2.py --input path_to_image/image.png --output path_to_save_image/image_name.png
```

Puedes probar con diferentes imagenes de diferentes formatos, solo debes asegurarte que tus imagenes esten en la carpeta con el nombre correcto