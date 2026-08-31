"""G2P: text -> ARPABET phone sequence via espeak-ng (bundled DLL) + IPA mapping.

Runs under the 3.11 env (phonemizer + espeakng-loader are Kokoro dependencies).

Note: phonemizer (espeak-ng backend) returns phonemes per WORD, concatenated
without separators inside the word (e.g. 'ðə kwɪk ...'). We parse each word by
greedy longest-prefix matching against the IPA table.
"""

from __future__ import annotations

import re

# IPA -> ARPABET (39-phone inventory used for SpeechOcean762). Longest-first lookup.
IPA_TO_ARPABET: dict[str, str] = {
    # vowels
    "iː": "IY", "i": "IY", "ɪ": "IH", "eɪ": "EY", "e": "EY", "ɛ": "EH",
    "æ": "AE", "ɑː": "AA", "ɑ": "AA", "ɒ": "AA", "ɔː": "AO", "ɔ": "AO",
    "oʊ": "OW", "o": "OW", "ʊ": "UH", "uː": "UW", "u": "UW", "ʌ": "AH",
    "ə": "AH", "ɐ": "AH", "ɜː": "ER", "ɜ": "ER", "ɝ": "ER", "ɚ": "ER",
    "aɪ": "AY", "aʊ": "AW", "ɔɪ": "OY",
    # reduced / central vowels
    "ᵻ": "AH", "ᵻ̞": "AH", "ɨ": "AH",
    # consonants
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G",
    "tʃ": "CH", "dʒ": "JH", "f": "F", "v": "V", "θ": "TH", "ð": "DH",
    "s": "S", "z": "Z", "ʃ": "SH", "ʒ": "ZH", "h": "HH",
    "m": "M", "n": "N", "ŋ": "NG", "l": "L", "ɹ": "R", "r": "R",
    "w": "W", "j": "Y", "ɾ": "T",
    # non-phonetic: dropped
    "ʔ": None,
    # syllabic consonants
    "ɹ̩": "ER", "n̩": "AH", "m̩": "AH", "l̩": "AH", "ŋ̩": "AH",
}

_STRESS_RE = re.compile(r"[ˈˌ']")
_KEYS = sorted(IPA_TO_ARPABET, key=len, reverse=True)

_phonemizer_ready = False
_cache: dict[str, list[str]] = {}
_cache_words: dict[str, list[list[str]]] = {}
_seen_unknown: set[str] = set()


def _ensure_phonemizer():
    global _phonemizer_ready
    if _phonemizer_ready:
        return
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    espeakng_loader.make_library_available()
    _phonemizer_ready = True


def _parse_word(tok: str) -> list[str]:
    """Greedy longest-prefix IPA -> ARPABET parse of one word's phoneme string."""
    out: list[str] = []
    while tok:
        for k in _KEYS:
            if tok.startswith(k):
                mapped = IPA_TO_ARPABET[k]
                if mapped:
                    out.append(mapped)
                tok = tok[len(k):]
                break
        else:
            _seen_unknown.add(tok[0])
            tok = tok[1:]
    return out


def text_to_arpabet_words(text: str) -> list[list[str]]:
    """Text -> ARPABET phones per word (word boundaries preserved)."""
    if not text:
        return []
    if text in _cache_words:
        return _cache_words[text]

    from phonemizer import phonemize

    _ensure_phonemizer()
    # lowercase to avoid acronym misreads (e.g. "IT" -> "eye-tee")
    ipa = phonemize(text.lower(), language="en-us").strip()
    out = [_parse_word(_STRESS_RE.sub("", w)) for w in ipa.split()]
    _cache_words[text] = out
    return out


def text_to_arpabet(text: str) -> list[str]:
    """Text -> stripped ARPABET phone list."""
    out: list[str] = []
    for w in text_to_arpabet_words(text):
        out.extend(w)
    return out


if __name__ == "__main__":
    for s in ("The quick brown fox jumps over the lazy dog", "Hello world"):
        print(s, "->", text_to_arpabet(s))
