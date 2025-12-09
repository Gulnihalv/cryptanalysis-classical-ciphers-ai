import os
from pathlib import Path

IS_COLAB = os.path.exists('/content/drive')

if IS_COLAB:
    ROOT_DIR = Path('/content/drive/MyDrive/Crypto-Project')
else:
    # Yerel bilgisayar (Repo'nun kök dizini)
    ROOT_DIR = Path(__file__).parent.parent 

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# MODEL_DIR = ROOT_DIR / "models_saved"