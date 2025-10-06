import argparse
import re
import requests
import json
from utils import  read_warc_file, read_wet_file
from datasets import load_dataset
from typing import Set, Dict
import string
from bs4 import BeautifulSoup
import langdetect
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open('./bad_word_list.txt', 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)


def html_to_text(html) -> str:
    """Converts HTML content to plain text..
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    
    try:
        # Decode bytes to string if necessary
        if isinstance(html, bytes):
            html = html.decode('utf-8', errors='ignore')
        
        # Use BeautifulSoup for robust HTML parsing
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean up whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error converting HTML to text: {e}")
        return ""

def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    # Replace US social security numbers (XXX-XX-XXXX format)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Replace phone numbers (various formats)
    text = re.sub(r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    
    # Replace credit card numbers (16 digits, possibly with spaces/dashes)
    text = re.sub(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', '[CREDIT_CARD]', text)
    
    # Replace URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    
    # Replace IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_ADDRESS]', text)
    
    return text
    

def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in string.whitespace)
    
    # Remove very short lines and lines with poor alphanumeric content
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if len(stripped_line) >= 10:
            alpha_count = sum(1 for char in stripped_line if char.isalnum())
            if alpha_count / len(stripped_line) > 0.3:  # At least 30% alphanumeric
                cleaned_lines.append(stripped_line)
    
    return ' '.join(cleaned_lines).strip()


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
    """
    # Get bad words list
    bad_words = retrieve_bad_words()
    
    text_lower = text.lower()
    
    # Check for bad words
    if any(bad_word in text_lower for bad_word in bad_words):
        return False
    
    # Check for sentence structure
    if not any(char in text for char in '.!?'):
        return False
    
    # Check for excessive capitalization
    upper_ratio = sum(1 for char in text if char.isupper()) / len(text)
    if upper_ratio > 0.5:
        return False
    
    # Check for excessive special characters
    special_ratio = sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text)
    if special_ratio > 0.3:
        return False
    
    return True



def is_english_text(text: str) -> bool:
    """Detects if text is primarily in English based on character distribution.
    Args:
        text (str): Text to analyze
    Returns:
        bool: True if text is primarily English, False otherwise
    """
    if not text:
        return False
    
    try:
        # Use langdetect library for language detection
        language = detect(text)
        return language == 'en'
    except:
        return False
    

def deduplicate_texts(texts: list[str]) -> list[str]:
    """Deduplicates text by removing duplicate sentences.
    Args:
        texts (list[str]): List of text strings to deduplicate.
    Returns:
        list[str]: Deduplicated list of texts. Implemented a simple Jaccard similarity based deduplication.
    """
    if not texts:
        return []
    
    # Use TF-IDF and cosine similarity for better deduplication
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    try:
        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicates using similarity threshold
        seen_indices = set()
        unique_texts = []
        
        for i in range(len(texts)):
            if i in seen_indices:
                continue
                
            unique_texts.append(texts[i])
            seen_indices.add(i)
            
            # Mark similar texts as duplicates
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i][j] > 0.8:  # 80% similarity threshold
                    seen_indices.add(j)
        
        return unique_texts
        
    except Exception as e:
        # Fallback to simple deduplication if TF-IDF fails
        print(f"TF-IDF deduplication failed, using fallback: {e}")
        seen = set()
        unique_texts = []
        for text in texts:
            if text and text not in seen:
                seen.add(text)
                unique_texts.append(text)
        return unique_texts



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str,  default = '', help = 'Specify the path for your warc file.')
    parser.add_argument('--dfname', type = str,  default = '', help = 'Specify the path where you stored topic_dataset.json')
    parser.add_argument('--num_records', type = int,  default=30, help = 'Specify the number of records you want to parse (only used for debugging with smaller sets)')
    parser.add_argument('--output', type = str,  default='cleaned_documents.txt', help = 'Output file for cleaned text documents')
    # parser.add_argument('--wet_name', type = str, default = '', help = 'Specify the path for your wet file.')
    args = parser.parse_args()

    if args.fname:
        seen = 0
        passes = 0

        with open(args.output, 'w', encoding='utf-8') as output_file:
            for url, html_text in read_warc_file(args.fname, args.num_records):
                seen += 1
                # print("Before HTML to text: ", str(html_text))
                text = html_to_text(html_text)
                # print("\n\n\nAfter HTML to text: ", text)
                cleaned_text = clean_text(text)
                # print("After cleaning: ", cleaned_text)
                cleaned_nopii_text = replace_pii(cleaned_text)
                # print("After PII removal: ", cleaned_nopii_text)
                passes_check = heuristic_quality_filter(cleaned_nopii_text)
                is_english = is_english_text(cleaned_nopii_text)
                print(url)
                print("Passes heuristic quality filter:", passes_check)
                print("Is English text:", is_english)
                if passes_check and is_english:
                    passes += 1
                    # Replace newlines with spaces to keep each document on one line
                    single_line_text = cleaned_nopii_text.replace('\n', ' ').replace('\r', ' ').strip()
                    output_file.write(single_line_text + '\n')
                    print("Saved cleaned English document to output file")
                elif passes_check and not is_english:
                    print("Document filtered out: not English")

        print(f"{passes} passed out of {seen} records processed.")
        print(f"Cleaned documents saved to: {args.output}")

    if args.dfname:
        with open(args.dfname, 'r') as f:
            raw_texts = json.load(f)
        raw_texts = [item['text'] for item in raw_texts['data']]
        deduplicated_texts = deduplicate_texts(raw_texts)
        print(f"{len(deduplicated_texts)} deduplicated out of {len(raw_texts)} records processed.")
    else:
        print("Usage: python homework.py --fname data.warc")