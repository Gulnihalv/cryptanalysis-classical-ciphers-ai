import sys
import os
import re
import logging
from collections import Counter
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
except ImportError:
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TurkishTextCleaner:
    # bu kısım makaleden alındı
    EXPECTED_FREQUENCIES = {
        'a': 11.82, 'e': 9.00, 'i': 8.34, 'n': 7.29, 'r': 6.98, 'l': 6.07,
        'ı': 5.12, 'k': 4.70, 'd': 4.63, 'm': 3.71, 'y': 3.42, 'u': 3.29,
        't': 3.27, 's': 3.03, 'b': 2.76, 'o': 2.47, 'ü': 1.97, 'ş': 1.83,
        'z': 1.51, 'g': 1.32, 'ç': 1.19, 'h': 1.11, 'ğ': 1.07, 'v': 1.00,
        'c': 0.97, 'ö': 0.86, 'p': 0.84, 'f': 0.43, 'j': 0.03
    }

    VALID_CHARS = set("abcçdefgğhıijklmnoöprsştuüvyz ")

    def __init__(self):
        self.abbreviations = [
            r'\bvs\.', r'\bvb\.', r'\bvd\.', r'\byy\.', r'\bs\.', 
            r'\bbkz\.', r'\börn\.', r'\byak\.', r'\bdoğ\.', r'\böl\.',
            r'\bdr\.', r'\bprof\.'
        ]

    def clean_text(self, text: str) -> str:
        # Structural Wiki Cleanup
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL) # Infoboxes
        text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)       # HTML tags
        
        # Reference Cleanup
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[(?:kaynak|citation).*?\]', '', text, flags=re.I)
        
        # Wiki Media/Links
        text = re.sub(r'\[\[(?:Dosya|File|Resim|Kategori):.*?\]\]', '', text, flags=re.I)
        text = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'\2', text)
        text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)

        text = re.sub(r'\b(d|ö)\.?\s*\d{3,4}\b', '', text)
        text = re.sub(r'\b\d{3,4}[\'’]?(?:ler|lar|den|dan|te|ta)\b', '', text)
        text = re.sub(r'\b\d{3,4}\s*[-–]\s*\d{3,4}\b', '', text)
        
        for abbrev in self.abbreviations:
            text = re.sub(abbrev, '', text, flags=re.I)

        text = text.replace('\n', ' ')
        text = re.sub(r'\b[A-ZÇĞİÖŞÜ]{3,}\b', '', text) 
        text = text.replace('İ', 'i').replace('I', 'ı').lower()
        text = re.sub(r'[«»""\'\"\\(){}\[\]]', '', text)

        mapping = {'â': 'a', 'î': 'i', 'û': 'u', 'ê': 'e', 'q': 'k', 'w': 'v', 'x': 'ks'}
        for src, dst in mapping.items():
            text = text.replace(src, dst)
        
        text = re.sub(r'[^abcçdefgğhıijklmnoöprsştuüvyz ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def is_high_quality_segment(self, text: str, min_len: int = 20) -> bool:
        if len(text) < min_len:
            return False

        valid_count = sum(1 for c in text if c in self.VALID_CHARS)
        ratio = valid_count / len(text)
        
        return ratio > 0.90

    def analyze_frequency(self, text: str) -> None:
        # Filter only alphabetic chars for statistics
        chars = [c for c in text if c.isalpha()]
        total = len(chars)
        
        if total == 0:
            logger.warning("No characters found for analysis.")
            return

        freq = Counter(chars)
        
        logger.info("\n" + "="*55)
        logger.info(f"{'CHAR':<6} {'COUNT':<10} {'ACTUAL %':<10} {'EXPECTED %':<12} {'DIFF'}")
        logger.info("-" * 55)

        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        total_diff = 0.0
        
        for char, count in sorted_chars:
            if char not in self.EXPECTED_FREQUENCIES:
                continue
                
            actual_pct = (count / total) * 100
            expected_pct = self.EXPECTED_FREQUENCIES[char]
            diff = actual_pct - expected_pct
            total_diff += abs(diff)
            
            flag = "!" if abs(diff) > 2.0 else ""
            
            if actual_pct > 1.0: 
                print(f"{char:<6} {count:<10,} {actual_pct:<10.2f} {expected_pct:<12.2f} {diff:+.2f} {flag}")

        logger.info("-" * 55)
        logger.info(f"Average Deviation: {total_diff / len(self.EXPECTED_FREQUENCIES):.2f}%")
        
        if total_diff / len(self.EXPECTED_FREQUENCIES) < 1.0:
            logger.info("Data distribution closely matches standard Turkish.")
        else:
            logger.warning("Data distribution shows significant deviation.")
        logger.info("="*55 + "\n")

    def process_file(self, input_path: Path, output_path: Path):
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        logger.info(f"Reading raw data from: {input_path}")
        
        with open(input_path, "r", encoding='utf-8') as f:
            content = f.read()
            
        logger.info(f"Original size: {len(content):,} chars")
        
        cleaned_content = self.clean_text(content)
        logger.info(f"Cleaned size: {len(cleaned_content):,} chars")
        
        self.analyze_frequency(cleaned_content)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
            
        logger.info(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    cleaner = TurkishTextCleaner()
    
    # raw_wiki_file = RAW_DATA_DIR / "wikipedia_tr.txt"
    # out_wiki_file = PROCESSED_DATA_DIR / "wiki_processed_tr.txt"
    
    # #cleaner.process_file(raw_wiki_file, out_wiki_file)

    # raw_book_file = RAW_DATA_DIR / "book_dataset.txt"
    # out_book_file = PROCESSED_DATA_DIR / "books_processed.txt"

    # #cleaner.process_file(raw_book_file, out_book_file)

    # raw_news_file = RAW_DATA_DIR / "newscor_test.txt"
    # out_news_file = PROCESSED_DATA_DIR / "newscor_processed.txt"

    # #cleaner.process_file(raw_news_file, out_news_file)

    raw_news_file = RAW_DATA_DIR / "wiki_news.txt"
    out_news_file = PROCESSED_DATA_DIR / "wiki_news_processed.txt"

    cleaner.process_file(raw_news_file, out_news_file)