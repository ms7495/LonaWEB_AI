# utils/file_utils.py - Simplified file utilities
import hashlib
import re
from pathlib import Path


def validate_file(file_path: str, max_size_mb: int = 200) -> tuple[bool, str]:
    """Validate uploaded file - from your validate_file function"""

    file_path = Path(file_path)

    if not file_path.exists():
        return False, "File does not exist"

    # Check size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"

    # Check extension
    ext = file_path.suffix.lower()
    supported_exts = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv']
    if ext not in supported_exts:
        return False, f"Unsupported file type: {ext}"

    return True, "File is valid"


def get_file_hash(file_path: str) -> str:
    """Get file hash for deduplication"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove problematic characters
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return clean_name
