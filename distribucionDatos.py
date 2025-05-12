import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ruta de entrada
INPUT_DIR = Path("categorias")
# Ruta de salida
OUTPUT_DIR = Path("dataset")
# Proporciones
TEST_RATIO = 0.2
VAL_RATIO = 0.2  # Sobre el 80% restante (20% de 80% = 16%)

# Categorías
CATEGORIES = ["AD", "CN", "EMCI", "LMCI"]

def split_and_copy_images():
    for category in CATEGORIES:
        print(f"Procesando categoría: {category}")
        
        category_path = INPUT_DIR / category
        images = list(category_path.glob("*.*"))
        images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        # Dividir en test y restante
        trainval_imgs, test_imgs = train_test_split(images, test_size=TEST_RATIO, random_state=42)
        # Dividir restante en train y val
        train_imgs, val_imgs = train_test_split(trainval_imgs, test_size=VAL_RATIO, random_state=42)

        # Copiar imágenes a sus respectivas carpetas
        copy_images(train_imgs, category, "train")
        copy_images(val_imgs, category, "val")
        copy_images(test_imgs, category, "test")

    print("División completada.")

def copy_images(images, category, subset):
    subset_dir = OUTPUT_DIR / subset / category
    subset_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        dest = subset_dir / img_path.name
        shutil.copy2(img_path, dest)

if __name__ == "__main__":
    split_and_copy_images()
