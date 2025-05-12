import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

# Ruta principal donde están las carpetas de categorías
BASE_DIR = Path("categorias")

# Categorías a procesar
CATEGORIES = ["AD", "CN", "EMCI", "LMCI"]

# Formatos válidos de imagen
VALID_EXTS = [".jpg", ".jpeg", ".png"]

def rename_images(category_path, category_name):
    image_files = [f for f in sorted(category_path.iterdir()) if f.suffix.lower() in VALID_EXTS]
    
    for idx, image_path in enumerate(image_files, 1):
        new_name = f"{category_name}_{idx:05d}{image_path.suffix.lower()}"
        new_path = category_path / new_name
        os.rename(image_path, new_path)

    print(f"[{category_name}] Renombradas {len(image_files)} imágenes.")

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        for category in CATEGORIES:
            category_path = BASE_DIR / category
            if category_path.exists() and category_path.is_dir():
                executor.submit(rename_images, category_path, category)

if __name__ == "__main__":
    main()
