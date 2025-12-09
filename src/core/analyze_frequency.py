import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collections import Counter
import json
from config.paths import PROCESSED_DATA_DIR

def analyze_unigrams(text):
    """
    Analyze character (unigram) frequencies
    """
    # Count only letters
    chars = [c for c in text if c.isalpha()]
    total = len(chars)
    
    freq = Counter(chars)
    
    # Calculate percentages
    freq_percent = {char: (count / total) * 100 
                    for char, count in freq.items()}
    
    # Sort by frequency (descending)
    sorted_freq = sorted(freq_percent.items(), 
                        key=lambda x: x[1], 
                        reverse=True)
    
    return sorted_freq, total


def analyze_bigrams(text):
    """
    Analyze bigram (2-character) frequencies
    """
    # Extract only letters
    chars = [c for c in text if c.isalpha()]
    
    # Create bigrams
    bigrams = [''.join(chars[i:i+2]) for i in range(len(chars) - 1)]
    total = len(bigrams)
    
    freq = Counter(bigrams)
    
    # Calculate percentages
    freq_percent = {bigram: (count / total) * 100 
                    for bigram, count in freq.items()}
    
    # Sort by frequency (descending)
    sorted_freq = sorted(freq_percent.items(), 
                        key=lambda x: x[1], 
                        reverse=True)
    
    return sorted_freq, total


def analyze_trigrams(text):
    """
    Analyze trigram (3-character) frequencies
    """
    # Extract only letters
    chars = [c for c in text if c.isalpha()]
    
    # Create trigrams
    trigrams = [''.join(chars[i:i+3]) for i in range(len(chars) - 2)]
    total = len(trigrams)
    
    freq = Counter(trigrams)
    
    # Calculate percentages
    freq_percent = {trigram: (count / total) * 100 
                    for trigram, count in freq.items()}
    
    # Sort by frequency (descending)
    sorted_freq = sorted(freq_percent.items(), 
                        key=lambda x: x[1], 
                        reverse=True)
    
    return sorted_freq, total


def print_unigram_table(unigrams, total_chars):
    """
    Print unigram frequency table
    """
    print("\n" + "=" * 70)
    print("📊 TURKISH CHARACTER FREQUENCIES (Unigrams)")
    print("=" * 70)
    print(f"Total characters analyzed: {total_chars:,}\n")
    
    # Print header
    print(f"{'No':<4} {'Char':<6} {'Count':<12} {'Frequency %':<15}")
    print("-" * 70)
    
    # Print all characters
    for i, (char, freq) in enumerate(unigrams, 1):
        count = int((freq / 100) * total_chars)
        print(f"{i:<4} {char.upper():<6} {count:<12,} {freq:<15.4f}")
    
    print("=" * 70)


def print_bigram_table(bigrams, total_bigrams, top_n=50):
    """
    Print bigram frequency table
    """
    print("\n" + "=" * 70)
    print(f"📊 TURKISH BIGRAM FREQUENCIES (Top {top_n})")
    print("=" * 70)
    print(f"Total bigrams analyzed: {total_bigrams:,}\n")
    
    # Split into two columns for compact display
    half = top_n // 2
    
    print(f"{'No':<4} {'Bigram':<8} {'Freq %':<12} | {'No':<4} {'Bigram':<8} {'Freq %':<12}")
    print("-" * 70)
    
    for i in range(half):
        left_idx = i
        right_idx = i + half
        
        # Left column
        left_no = left_idx + 1
        left_bg, left_freq = bigrams[left_idx]
        left_str = f"{left_no:<4} {left_bg.upper():<8} {left_freq:<12.5f}"
        
        # Right column
        if right_idx < len(bigrams):
            right_no = right_idx + 1
            right_bg, right_freq = bigrams[right_idx]
            right_str = f"{right_no:<4} {right_bg.upper():<8} {right_freq:<12.5f}"
        else:
            right_str = ""
        
        print(f"{left_str} | {right_str}")
    
    print("=" * 70)


def print_trigram_table(trigrams, total_trigrams, top_n=50):
    """
    Print trigram frequency table
    """
    print("\n" + "=" * 70)
    print(f"📊 TURKISH TRIGRAM FREQUENCIES (Top {top_n})")
    print("=" * 70)
    print(f"Total trigrams analyzed: {total_trigrams:,}\n")
    
    # Split into two columns
    half = top_n // 2
    
    print(f"{'No':<4} {'Trigram':<10} {'Freq %':<12} | {'No':<4} {'Trigram':<10} {'Freq %':<12}")
    print("-" * 70)
    
    for i in range(half):
        left_idx = i
        right_idx = i + half
        
        # Left column
        left_no = left_idx + 1
        left_tg, left_freq = trigrams[left_idx]
        left_str = f"{left_no:<4} {left_tg.upper():<10} {left_freq:<12.5f}"
        
        # Right column
        if right_idx < len(trigrams):
            right_no = right_idx + 1
            right_tg, right_freq = trigrams[right_idx]
            right_str = f"{right_no:<4} {right_tg.upper():<10} {right_freq:<12.5f}"
        else:
            right_str = ""
        
        print(f"{left_str} | {right_str}")
    
    print("=" * 70)


def save_frequencies_to_json(unigrams, bigrams, trigrams, output_file):
    """
    Save all frequencies to JSON file
    """
    data = {
        'unigrams': {char: freq for char, freq in unigrams},
        'bigrams': {bg: freq for bg, freq in bigrams[:100]},  # Top 100
        'trigrams': {tg: freq for tg, freq in trigrams[:100]}  # Top 100
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Frequencies saved to: {output_file}")


def compare_with_reference(unigrams):
    """
    Compare with reference Turkish frequencies
    """
    # Reference frequencies from literature
    reference = {
        'a': 11.92, 'e': 8.91, 'i': 8.60, 'n': 7.26,
        'r': 6.95, 'l': 5.75, 'ı': 5.11, 't': 3.31,
        'k': 4.68, 'd': 4.43, 'm': 3.75, 'y': 3.34,
        'u': 3.23, 's': 3.00, 'o': 2.89, 'b': 2.78,
        'ü': 1.97, 'ş': 1.83, 'z': 1.51, 'g': 1.32,
        'ç': 1.19, 'h': 1.11, 'ğ': 1.07, 'v': 1.00,
        'c': 0.97, 'ö': 0.86, 'p': 0.84, 'f': 0.43,
        'j': 0.03
    }
    
    print("\n" + "=" * 80)
    print("📊 COMPARISON WITH REFERENCE FREQUENCIES")
    print("=" * 80)
    
    print(f"{'Char':<6} {'Our Freq %':<15} {'Reference %':<15} {'Diff':<10} {'Status'}")
    print("-" * 80)
    
    unigram_dict = dict(unigrams)
    
    for char in sorted(reference.keys()):
        our_freq = unigram_dict.get(char, 0)
        ref_freq = reference[char]
        diff = our_freq - ref_freq
        
        # Status indicator
        if abs(diff) < 0.5:
            status = "✅ Excellent"
        elif abs(diff) < 1.0:
            status = "✓ Good"
        elif abs(diff) < 2.0:
            status = "⚠️ Fair"
        else:
            status = "❌ Check"
        
        print(f"{char.upper():<6} {our_freq:<15.2f} {ref_freq:<15.2f} {diff:+<10.2f} {status}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 70)
    print("TURKISH TEXT FREQUENCY ANALYSIS")
    print("=" * 70)
    
    # Load processed text
    input_file = PROCESSED_DATA_DIR / "wiki_processed_tr2.txt"
    print(f"\nLoading: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text size: {len(text):,} characters")
    
    # Analyze unigrams
    print("\n🔍 Analyzing unigrams...")
    unigrams, total_chars = analyze_unigrams(text)
    print_unigram_table(unigrams, total_chars)
    
    # Analyze bigrams
    print("\n🔍 Analyzing bigrams...")
    bigrams, total_bigrams = analyze_bigrams(text)
    print_bigram_table(bigrams, total_bigrams, top_n=50)
    
    # Analyze trigrams
    print("\n🔍 Analyzing trigrams...")
    trigrams, total_trigrams = analyze_trigrams(text)
    print_trigram_table(trigrams, total_trigrams, top_n=50)
    
    # Compare with reference
    compare_with_reference(unigrams)
    
    # Save to JSON
    output_json = PROCESSED_DATA_DIR / "turkish_frequencies.json"
    save_frequencies_to_json(unigrams, bigrams, trigrams, output_json)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📈 SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total characters: {total_chars:,}")
    print(f"Unique characters: {len(unigrams)}")
    print(f"Total bigrams: {total_bigrams:,}")
    print(f"Unique bigrams: {len(bigrams)}")
    print(f"Total trigrams: {total_trigrams:,}")
    print(f"Unique trigrams: {len(trigrams)}")
    print(f"\nTop 5 characters: {', '.join(c.upper() for c, _ in unigrams[:5])}")
    print(f"Top 5 bigrams: {', '.join(bg.upper() for bg, _ in bigrams[:5])}")
    print(f"Top 5 trigrams: {', '.join(tg.upper() for tg, _ in trigrams[:5])}")
    print("=" * 70)