import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset
from config.paths import RAW_DATA_DIR

def download_tr_wikipedia():
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.tr",
        split="train[:1%]"
    )

    ds_text = ds['text']

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DATA_DIR / f"wikipedia_tr.txt"

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in ds_text:
            file.write(line)
    print(f"Saved → {output_file}")

if __name__ == "__main__":
    download_tr_wikipedia()

