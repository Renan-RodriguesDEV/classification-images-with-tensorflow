# cleanup_dataset.py
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")

for img_path in DATASET_DIR.rglob("*"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
        continue
    try:
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")   # força RGB — resolve canais 2, 4, L+A etc
            rgb.save(img_path)         # sobrescreve no lugar
    except Exception as e:
        print(f"Removendo imagem inválida: {img_path} → {e}")
        img_path.unlink()              # deleta imagens que não abrem de jeito nenhum