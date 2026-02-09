"""Simhash helpers for structure similarity checks."""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import regex as re
from simhash import Simhash, SimhashIndex


class ComputeSimhash:
    """Compute Simhash values for structure keys.

    The implementation mirrors the original project logic to preserve behavior.
    """

    def __init__(self) -> None:
        self.simhash_index: Optional[SimhashIndex] = None

    def hashing(
        self,
        document: str,
        tokenization: str = "punctuation",
        ignore_punctuation: bool = False,
        lowercase: bool = True,
    ) -> Simhash:
        """Compute a Simhash for the given document.

        Args:
            document: Input string.
            tokenization: Tokenization mode ("punctuation", "character", "space").
            ignore_punctuation: Whether to strip punctuation before hashing.
            lowercase: Whether to lowercase the input.

        Returns:
            A Simhash object (f=128).
        """
        punctuation_regex = re.compile(r"\p{P}-[-_=:]|[/; ]")

        token_size = [1, 1, 1]
        window_size = [100, 2, 1]
        index = 0
        tokens: list[bytes] = []
        for sub_document in document.split("|"):
            if lowercase:
                sub_document = sub_document.lower()

            if ignore_punctuation:
                sub_document = punctuation_regex.sub("", sub_document)

            if tokenization == "character":
                sub_tokens = [
                    str.encode(sub_document[i : i + window_size[index]])
                    for i in range(len(sub_document) - window_size[index])
                ]
            elif tokenization == "punctuation":
                split_tokens = punctuation_regex.split(sub_document)
                sub_tokens = [
                    str.encode(" ".join(split_tokens[i : i + window_size[index]]))
                    for i in range(len(split_tokens))
                ]
            elif tokenization == "space":
                split_tokens = sub_document.split(" ")
                sub_tokens = [
                    str.encode(" ".join(split_tokens[i : i + window_size[index]]))
                    for i in range(len(split_tokens) - window_size[index])
                ]
            else:
                raise ValueError(f"Unrecognized tokenization parameter {tokenization}")

            for _ in range(token_size[index] - len(sub_tokens)):
                sub_tokens.append(str.encode(""))
            tokens.extend(sub_tokens)
            index += 1

        tokens = [hashlib.md5(token).hexdigest() for token in tokens]
        return Simhash(tokens, f=128)

    def distance(self, hash1: Simhash, hash2: Simhash) -> int:
        """Return Simhash distance between two hashes."""
        return hash1.distance(hash2)

    def straight_hash(self, document: str) -> Simhash:
        """Return a direct Simhash of the input string."""
        return Simhash(document)

    def evaluate(self, line: str) -> Simhash:
        """Evaluate the hash for a line."""
        return self.hashing(line)

    def is_conflict(self, key: str, threshold: int) -> Optional[Iterable[str]]:
        """Check if key conflicts in index and add it if not.

        Args:
            key: Structure key string.
            threshold: Max distance to treat as duplicate.

        Returns:
            Iterable of duplicate keys or None.
        """
        if self.simhash_index is None:
            self.simhash_index = SimhashIndex([], k=threshold, f=128)
        hash_value = self.evaluate(key)
        duplicate = self.simhash_index.get_near_dups(hash_value)
        if not duplicate:
            self.simhash_index.add(key, hash_value)
            return None
        return duplicate

    def is_conflict_noadd(self, key: str, threshold: int) -> Optional[Iterable[str]]:
        """Check if key conflicts in index without adding.

        Args:
            key: Structure key string.
            threshold: Max distance to treat as duplicate.

        Returns:
            Iterable of duplicate keys or None.
        """
        if self.simhash_index is None:
            self.simhash_index = SimhashIndex([], k=threshold, f=128)
        hash_value = self.evaluate(key)
        duplicate = self.simhash_index.get_near_dups(hash_value)
        if not duplicate:
            return None
        return duplicate
