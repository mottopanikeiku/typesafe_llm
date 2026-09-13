"""Frequency-ranked lexical proposals; no model outputs or task answers as data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

DEFAULT_LEXICON = Path(__file__).resolve().parent / "data" / "english_words.json"
CANDIDATE_LIMIT = 1024
GROUP_SIZE = 128
WORD_END = re.compile(r"[A-Za-z]+$")


@dataclass(frozen=True)
class WordLexicon:
    words: tuple[str, ...]
    source: str
    sha256: str

    @classmethod
    def load(cls, path: Path) -> WordLexicon:
        raw = path.read_bytes()
        document = json.loads(raw)
        words = document.get("words") if isinstance(document, dict) else None
        if not isinstance(words, list) or not words:
            raise ValueError("Lexicon must be a JSON object with a nonempty words array.")
        if any(not isinstance(word, str) or not word.isascii() or not word.isalpha()
               or word != word.lower() for word in words):
            raise ValueError("Lexicon words must be nonempty lowercase ASCII letters only.")
        return cls(tuple(dict.fromkeys(words)), str(path.resolve()), hashlib.sha256(raw).hexdigest())

    def groups(self, prefix: str, vocabulary: dict[str, str | None]) -> dict[str, dict[str, str]]:
        """Propose suffixes of matching words in frequency order, without editing prefix."""
        match = WORD_END.search(prefix)
        stem = match.group() if match else ""
        upper = len(stem) > 1 and stem.isupper()
        capitalized = stem[:1].isupper()
        seen = set(vocabulary.values())
        groups: dict[str, dict[str, str]] = {}
        count = 0
        for word in self.words:
            candidate = word.upper() if upper else word.capitalize() if capitalized else word
            if not candidate.startswith(stem) or len(candidate) <= len(stem):
                continue
            suffix = candidate[len(stem):]
            if suffix in seen:
                continue
            label = f"WORD_{count:04d}"
            while label in vocabulary:
                label = "_" + label
            groups.setdefault(f"word_group_{count // GROUP_SIZE}", {})[label] = suffix
            seen.add(suffix)
            count += 1
            if count == CANDIDATE_LIMIT:
                break
        return groups
