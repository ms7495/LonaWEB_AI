import logging
import re
from typing import List

import unicodedata

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize('NFKD', text)

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    return text.strip()


def extract_sentences(text: str) -> List[str]:
    if not text:
        return []

    sentences = re.split(r'[.!?]+\s+', text)

    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


def truncate_text(text: str, max_length: int = 500, add_ellipsis: bool = True) -> str:
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]

    if add_ellipsis:
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]
        truncated += "..."

    return truncated


def remove_extra_whitespace(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r' +', ' ', text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    if not text:
        return []

    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, count in sorted_words[:max_keywords]]

    return keywords


def highlight_text(text: str, search_terms: List[str], highlight_tag: str = "**") -> str:
    if not text or not search_terms:
        return text

    highlighted_text = text

    for term in search_terms:
        if term:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f"{highlight_tag}{term}{highlight_tag}",
                highlighted_text
            )

    return highlighted_text


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            last_sentence = text.rfind('.', start, end)
            last_question = text.rfind('?', start, end)
            last_exclamation = text.rfind('!', start, end)

            sentence_end = max(last_sentence, last_question, last_exclamation)

            if sentence_end > start + chunk_size * 0.5:
                end = sentence_end + 1
            else:
                last_space = text.rfind(' ', start, end)
                if last_space > start + chunk_size * 0.5:
                    end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + chunk_size - overlap, end - overlap)

        if start >= len(text):
            break

    return chunks


def calculate_text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0

    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def format_text_for_display(text: str, max_width: int = 80) -> str:
    if not text:
        return ""

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)
