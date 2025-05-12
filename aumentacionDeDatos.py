import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from concurrent.futures import ThreadPoolExecutor

input_dir = "categorias/CN"
output_dir = "categorias/CN"  # O cambia a otra carpeta
target_total = 8800

existing_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
n_existing = len(existing_images)
n_to_generate = target_total - n_existing

# Transformaciones con ajuste de recorte
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
])

# Función que realiza la aumentación y guarda la imagen
def augment_and_save_image(image_path, idx):
    # Cargar imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplicar la transformacion
    augmented = transform(image=image)
    aug_img = augmented["image"]

    # Guardar la imagen aumentada
    base_name = os.path.basename(image_path)
    new_filename = f"aug_{idx}_{base_name}"
    output_path = os.path.join(output_dir, new_filename)
    Image.fromarray(aug_img).save(output_path)

# Función para realizar la aumentación en paralelo
def process_images_in_parallel():
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(n_to_generate):
            base_name = np.random.choice(existing_images)
            img_path = os.path.join(input_dir, base_name)

            # Ejecutar el proceso de aumentación y guardado de forma paralela
            futures.append(executor.submit(augment_and_save_image, img_path, i))

        # Esperar a que todos los procesos terminen
        for future in futures:
            future.result()  # Obtiene el resultado (esto bloquea hasta que cada tarea termine)

    print("¡Aumentación completada en paralelo!")

# Ejecutar el proceso
process_images_in_parallel()
