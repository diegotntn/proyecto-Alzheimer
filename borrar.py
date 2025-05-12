import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Ruta principal donde están las carpetas de categorías
BASE_DIR = Path("categorias")

# Categorías válidas
CATEGORIES = ["AD", "CN", "EMCI", "LMCI"]

# Formatos válidos de imagen
VALID_EXTS = [".jpg", ".jpeg", ".png"]

def delete_non_matching_files(category_path, category_name):
    image_files = [f for f in category_path.iterdir() if f.suffix.lower() in VALID_EXTS]

    deleted_count = 0
    pattern = re.compile(rf"^{category_name}_\d{{5}}\.(jpg|jpeg|png)$", re.IGNORECASE)

    for image_path in image_files:
        if not pattern.match(image_path.name):
            image_path.unlink()
            deleted_count += 1

    print(f"[{category_name}] Eliminados {deleted_count} archivos no válidos.")

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        for category in CATEGORIES:
            category_path = BASE_DIR / category
            if category_path.exists() and category_path.is_dir():
                executor.submit(delete_non_matching_files, category_path, category)

if __name__ == "__main__":
    main()
