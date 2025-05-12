import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

# Ruta a la carpeta que contiene las imágenes
image_folder = 'LMCI'

# Obtener lista de rutas de imágenes
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Función para procesar una imagen
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    return resized

def main():
    processed_images = []

    with ProcessPoolExecutor() as executor:
        future_to_path = {executor.submit(process_image, path): path for path in image_paths}
        for future in as_completed(future_to_path):
            try:
                processed_image = future.result()
                processed_images.append(processed_image)
            except Exception as exc:
                print(f"Error procesando {future_to_path[future]}: {exc}")

    print(f"Procesadas {len(processed_images)} imágenes.")

if __name__ == '__main__':
    freeze_support()  # Necesario en Windows si se va a congelar el script
    main()
