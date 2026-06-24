import re
from config import PLATE_REGEX

_pattern = re.compile(PLATE_REGEX)

def is_valid(plate_text):
    if not plate_text:
        return False
    cleaned = plate_text.replace(" ", "").replace("-", "").upper()
    return bool(_pattern.search(cleaned))

def clean(plate_text):
    cleaned = plate_text.replace(" ", "").replace("-", "").upper()
    match = _pattern.search(cleaned)
    if match:
        return match.group(0)
    return cleaned
