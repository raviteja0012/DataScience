"""Matching strategy implementations.

Provides exact, fuzzy (Levenshtein, Jaro-Winkler), and phonetic (Soundex,
Metaphone) matching algorithms.  Each strategy produces a normalised
similarity score in [0.0, 1.0].
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MatchingStrategy(ABC):
    """Base class for a field-level matching strategy."""

    name: str = "base"

    @abstractmethod
    def score(self, value_a: Any, value_b: Any) -> float:
        """Return a similarity score in [0.0, 1.0]."""
        ...

    def _normalise(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()


class ExactMatchStrategy(MatchingStrategy):
    """Binary exact-match comparison (case-insensitive, trimmed)."""

    name = "exact"

    def score(self, value_a: Any, value_b: Any) -> float:
        a = self._normalise(value_a)
        b = self._normalise(value_b)
        if not a or not b:
            return 0.0
        return 1.0 if a == b else 0.0


class LevenshteinStrategy(MatchingStrategy):
    """Normalised Levenshtein distance (edit distance / max_len)."""

    name = "levenshtein"

    def score(self, value_a: Any, value_b: Any) -> float:
        a = self._normalise(value_a)
        b = self._normalise(value_b)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        dist = self._levenshtein(a, b)
        max_len = max(len(a), len(b))
        return round(1.0 - dist / max_len, 4)

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return LevenshteinStrategy._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]


class JaroWinklerStrategy(MatchingStrategy):
    """Jaro-Winkler similarity — favours strings that share a common prefix."""

    name = "jaro_winkler"

    def score(self, value_a: Any, value_b: Any) -> float:
        a = self._normalise(value_a)
        b = self._normalise(value_b)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return round(self._jaro_winkler(a, b), 4)

    @staticmethod
    def _jaro(s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        max_dist = max(len1, len2) // 2 - 1
        if max_dist < 0:
            max_dist = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - max_dist)
            end = min(i + max_dist + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (
            matches / len1
            + matches / len2
            + (matches - transpositions / 2) / matches
        ) / 3.0
        return jaro

    def _jaro_winkler(self, s1: str, s2: str, p: float = 0.1) -> float:
        jaro = self._jaro(s1, s2)
        prefix_len = 0
        for i in range(min(4, len(s1), len(s2))):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break
        return jaro + prefix_len * p * (1 - jaro)


class SoundexStrategy(MatchingStrategy):
    """American Soundex phonetic matching."""

    name = "soundex"

    def score(self, value_a: Any, value_b: Any) -> float:
        a = self._normalise(value_a)
        b = self._normalise(value_b)
        if not a or not b:
            return 0.0
        sa = self._soundex(a)
        sb = self._soundex(b)
        if sa == sb:
            return 1.0
        # Partial credit for close Soundex codes
        matching = sum(c1 == c2 for c1, c2 in zip(sa, sb))
        return round(matching / 4.0, 4)

    @staticmethod
    def _soundex(word: str) -> str:
        if not word:
            return "0000"
        word = word.upper()
        code_map = {
            "B": "1", "F": "1", "P": "1", "V": "1",
            "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2",
            "S": "2", "X": "2", "Z": "2",
            "D": "3", "T": "3",
            "L": "4",
            "M": "5", "N": "5",
            "R": "6",
        }
        coded = [word[0]]
        prev = code_map.get(word[0], "0")
        for ch in word[1:]:
            c = code_map.get(ch, "0")
            if c != "0" and c != prev:
                coded.append(c)
            prev = c if c != "0" else prev
        result = "".join(coded)
        return (result + "0000")[:4]


class MetaphoneStrategy(MatchingStrategy):
    """Simplified Metaphone phonetic matching."""

    name = "metaphone"

    def score(self, value_a: Any, value_b: Any) -> float:
        a = self._normalise(value_a)
        b = self._normalise(value_b)
        if not a or not b:
            return 0.0
        ma = self._metaphone(a)
        mb = self._metaphone(b)
        if ma == mb:
            return 1.0
        # Partial credit via Levenshtein on the metaphone codes
        dist = LevenshteinStrategy._levenshtein(ma, mb)
        max_len = max(len(ma), len(mb))
        if max_len == 0:
            return 1.0
        return round(max(0.0, 1.0 - dist / max_len), 4)

    @staticmethod
    def _metaphone(word: str) -> str:
        """Simplified Metaphone encoding."""
        if not word:
            return ""
        word = word.upper()
        # Drop initial silent consonant pairs
        for prefix in ("AE", "GN", "KN", "PN", "WR"):
            if word.startswith(prefix):
                word = word[1:]
                break

        result = []
        i = 0
        while i < len(word) and len(result) < 6:
            ch = word[i]
            if ch in "AEIOU":
                if i == 0:
                    result.append(ch)
            elif ch == "B":
                if i == 0 or word[i - 1] != "M":
                    result.append("B")
            elif ch == "C":
                if i + 1 < len(word) and word[i + 1] in "EIY":
                    result.append("S")
                else:
                    result.append("K")
            elif ch == "D":
                if i + 1 < len(word) and word[i + 1] in "GEI":
                    result.append("J")
                else:
                    result.append("T")
            elif ch in "FJ":
                result.append(ch)
            elif ch == "G":
                if i + 1 < len(word) and word[i + 1] in "EIY":
                    result.append("J")
                elif i == 0 or word[i - 1] not in "DG":
                    result.append("K")
            elif ch == "H":
                if i == 0 or word[i - 1] not in "AEIOU":
                    if i + 1 < len(word) and word[i + 1] in "AEIOU":
                        result.append("H")
            elif ch == "K":
                if i == 0 or word[i - 1] != "C":
                    result.append("K")
            elif ch == "L":
                result.append("L")
            elif ch in "MN":
                result.append(ch)
            elif ch == "P":
                if i + 1 < len(word) and word[i + 1] == "H":
                    result.append("F")
                    i += 1
                else:
                    result.append("P")
            elif ch == "Q":
                result.append("K")
            elif ch == "R":
                result.append("R")
            elif ch == "S":
                if i + 1 < len(word) and word[i + 1] == "H":
                    result.append("X")
                    i += 1
                elif i + 2 < len(word) and word[i:i+3] == "SIO":
                    result.append("X")
                else:
                    result.append("S")
            elif ch == "T":
                if i + 1 < len(word) and word[i + 1] == "H":
                    result.append("0")  # theta
                    i += 1
                else:
                    result.append("T")
            elif ch == "V":
                result.append("F")
            elif ch == "W":
                if i + 1 < len(word) and word[i + 1] in "AEIOU":
                    result.append("W")
            elif ch == "X":
                result.append("KS")
            elif ch == "Y":
                if i + 1 < len(word) and word[i + 1] in "AEIOU":
                    result.append("Y")
            elif ch == "Z":
                result.append("S")
            i += 1

        return "".join(result)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type[MatchingStrategy]] = {
    "exact": ExactMatchStrategy,
    "levenshtein": LevenshteinStrategy,
    "fuzzy": JaroWinklerStrategy,        # alias
    "jaro_winkler": JaroWinklerStrategy,
    "soundex": SoundexStrategy,
    "phonetic": MetaphoneStrategy,        # alias
    "metaphone": MetaphoneStrategy,
}


def get_strategy(name: str) -> MatchingStrategy:
    """Instantiate a matching strategy by name."""
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown matching strategy: {name}")
    return cls()
