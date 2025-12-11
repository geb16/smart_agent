

import re

PII_PATTERNS = [
    r"\b\d{3}-\d{3}-\d{4}\b",     # phone
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z.-]+\.[A-Z]{2,}\b",  # email
    r"\b\d{16}\b",                # card number
]

def scrub_output(text: str) -> str:
    cleaned = text
    for p in PII_PATTERNS:
        cleaned = re.sub(p, "[REDACTED]", cleaned)
    return cleaned

