import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
from config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

def clean_wikipedia_text(text):
    """
    Wikipedia-specific cleaning for Turkish
    """
    # Remove infobox templates
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    
    # Remove references [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove citation needed, kaynak belirtiniz, etc.
    text = re.sub(r'\[kaynak belirtiniz\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    
    # Remove categories
    text = re.sub(r'\[\[Kategori:.*?\]\]', '', text, flags=re.IGNORECASE)
    
    # Remove file/image references
    text = re.sub(r'\[\[Dosya:.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[Resim:.*?\]\]', '', text, flags=re.IGNORECASE)
    
    # Remove internal links [[...]]
    text = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'\2', text)  # [[link|text]] -> text
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)  # [[link]] -> link
    
    # Remove birth/death dates: "d. 1923", "ö. 1985"
    text = re.sub(r'\bd\.\s*\d{3,4}\b', '', text)
    text = re.sub(r'\bö\.\s*\d{3,4}\b', '', text)
    text = re.sub(r'\bd\s*\d{3,4}\b', '', text)
    text = re.sub(r'\bö\s*\d{3,4}\b', '', text)
    
    # Remove year ranges: "1923-1985", "1920'ler"
    text = re.sub(r'\b\d{3,4}\s*-\s*\d{3,4}\b', '', text)
    text = re.sub(r'\b\d{3,4}\'ler\b', '', text)
    text = re.sub(r'\b\d{3,4}\'lar\b', '', text)
    
    # Remove standalone years (careful - keep those in sentences)
    # Only remove if surrounded by spaces/punctuation
    text = re.sub(r'(?<![a-züğışöç])\d{3,4}(?![a-züğışöç])', '', text)
    text = re.sub(r'\b\d+\s*(km|m|cm|mm|kg|g|lt)\b', '', text)
    
    abbrevs = [
        r'\bvs\.',
        r'\bvb\.',  
        r'\bvd\.',  
        r'\byy\.',  
        r'\bs\.',   
        r'\bbkz\.', 
        r'\börn\.', 
        r'\byak\.', 
        r'\bdoğ\.', 
        r'\böl\.',  
    ]
    for abbrev in abbrevs:
        text = re.sub(abbrev, '', text, flags=re.IGNORECASE)
    
    # Remove lines starting with bullets/numbers
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        line = line.strip()
        # Skip if starts with bullet/number
        if re.match(r'^[\-\*•]\s', line):
            continue
        if re.match(r'^\d+\.\s', line):
            continue
        if len(line) < 20:
            continue
        filtered_lines.append(line)
    
    text = ' '.join(filtered_lines)
    
    # Remove sections that are all caps (likely headers)
    text = re.sub(r'\b[A-ZÇĞİÖŞÜ]{3,}\b', '', text)
    text = text.replace('I', 'ı').replace('İ', 'i').lower()

    text = re.sub(r'[«»""\'\"\\]', '', text)  # Remove quotes
    
    # Normalize special characters
    mapping = {
        'â': 'a', 'î': 'i', 'û': 'u', 'ê': 'e',
        'q': 'k', 'w': 'v', 'x': 'ks',
    }
    for source, dest in mapping.items():
        text = text.replace(source, dest)
    
    text = re.sub(r'\d+', '', text)
    
    # Remove non-Turkish characters
    text = re.sub(r'[^abcçdefgğhıijklmnoöprsştuüvyz .!?]', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def split_into_sentences(text, min_length=30, max_length=200):
    """
    Split text into clean sentences
    """
    # Split by sentence boundaries
    sentences = re.split(r'[.!?]+', text)
    
    # Filter and clean
    filtered = []
    for sent in sentences:
        sent = sent.strip()
        
        # Length filter
        if not (min_length <= len(sent) <= max_length):
            continue
        
        # Quality checks
        if not is_good_sentence(sent):
            continue
        
        filtered.append(sent)
    
    return filtered


def is_good_sentence(sentence):
    """
    Quality check for sentences
    """
    # Must have at least 5 words
    words = sentence.split()
    if len(words) < 5:
        return False
    
    # Must have at least 3 Turkish-specific characters
    turkish_chars = 'çğıiİöşü'
    turkish_count = sum(1 for c in sentence if c in turkish_chars)
    if turkish_count < 3:
        return False
    
    # Word diversity (avoid repetitive sentences)
    unique_words = len(set(words))
    if unique_words < len(words) * 0.6:  # At least 60% unique
        return False
    
    # Avoid sentences with too many single-letter words
    single_char_words = sum(1 for w in words if len(w) == 1)
    if single_char_words > len(words) * 0.3:
        return False
    
    # Must contain at least one verb indicator (very basic check)
    # Turkish verb endings: -yor, -miş, -dı, -di, etc.
    verb_patterns = ['yor', 'miş', 'mış', 'muş', 'müş', 'dı', 'di', 'du', 'dü']
    has_verb = any(pattern in sentence for pattern in verb_patterns)
    if not has_verb:
        return False
    
    return True


def analyze_character_frequency(text):
    """
    Analyze character frequency to detect anomalies
    """
    from collections import Counter
    
    # Count characters
    chars = [c for c in text if c.isalpha()]
    freq = Counter(chars)
    total = len(chars)
    
    print("\n📊 Character Frequency Analysis:")
    print("=" * 50)
    
    # Expected Turkish frequencies
    expected_freq = {
        'a': 11.92, 'e': 8.91, 'i': 8.60, 'n': 7.26,
        'r': 6.95, 'l': 5.75, 'ı': 5.11, 't': 3.31,
        'k': 4.68, 'd': 4.43, 'm': 3.75, 'y': 3.34,
        'u': 3.23, 's': 3.00, 'o': 2.89, 'b': 2.78,
    }
    
    # Compare top characters
    top_chars = freq.most_common(20)
    
    print(f"{'Char':<6} {'Count':<10} {'Freq %':<10} {'Expected %':<12} {'Diff'}")
    print("-" * 50)
    
    for char, count in top_chars:
        freq_pct = (count / total) * 100
        expected = expected_freq.get(char, 0)
        diff = freq_pct - expected
        
        # Flag anomalies
        flag = "⚠️" if abs(diff) > 2 else ""
        
        print(f"{char:<6} {count:<10,} {freq_pct:<10.2f} {expected:<12.2f} {diff:+.2f} {flag}")
    
    print("=" * 50)
    
    # Check for anomalies
    anomalies = []
    for char, count in freq.most_common(10):
        freq_pct = (count / total) * 100
        expected = expected_freq.get(char, 3.0)  # Default if not in dict
        if abs(freq_pct - expected) > 2:
            anomalies.append((char, freq_pct, expected))
    
    if anomalies:
        print("\n⚠️ Detected anomalies:")
        for char, actual, expected in anomalies:
            print(f"  '{char}': {actual:.1f}% (expected ~{expected:.1f}%)")
    else:
        print("\n✅ Character frequencies look normal!")


if __name__ == "__main__":
    print("=" * 60)
    print("Wikipedia Turkish Text Cleaner")
    print("=" * 60)
    
    # Read raw file
    raw_file_path = RAW_DATA_DIR / "wikipedia_tr.txt"
    print(f"\nReading: {raw_file_path}")
    
    with open(raw_file_path, "r", encoding='utf-8') as file:
        content = file.read()
    
    print(f"Original size: {len(content):,} characters")
    
    # Clean
    print("\nCleaning...")
    cleaned_text = clean_wikipedia_text(content)
    print(f"Cleaned size: {len(cleaned_text):,} characters")
    
    # Analyze frequency BEFORE splitting
    print("\n" + "=" * 60)
    print("Analyzing character frequencies...")
    analyze_character_frequency(cleaned_text)
    
    # Split into sentences
    print("\n" + "=" * 60)
    print("Splitting into sentences...")
    sentences = split_into_sentences(cleaned_text)
    print(f"Total sentences: {len(sentences):,}")
    
    if len(sentences) > 0:
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        print(f"Average sentence length: {avg_length:.1f} characters")
        
        # Show samples
        print("\n📝 Sample sentences:")
        for i, sent in enumerate(sentences[:5], 1):
            print(f"  {i}. {sent}")
    
    # Save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "wiki_processed_tr.txt"
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for sent in sentences:
            file.write(sent + '\n')
    
    print(f"\n✅ Saved → {output_file}")
    print(f"✅ Total sentences: {len(sentences):,}")
    print("=" * 60)