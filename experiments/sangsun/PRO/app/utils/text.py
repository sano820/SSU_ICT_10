import hashlib
import re


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_whitespace(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()
