"""Self-written procedural corpora (no downloads).

`build_corpus` expands a set of hand-written sentence templates about a small
harbour town into ~50KB of deterministic English-like prose.  `build_alt_corpus`
produces an equally sized but stylistically alien pseudo-code corpus, used to
train the "wrong corpus" draft model for the break-it experiment.
"""

from __future__ import annotations

import random

# --- hand-written raw material -------------------------------------------------

_NAMES = [
    "Mara", "Old Tennet", "Ilsa", "the archivist", "the ferryman",
    "Doran", "the lamplighter", "Petra", "the harbourmaster", "Wen",
]
_PLACES = [
    "the lighthouse", "the archive", "the tide pools", "the north pier",
    "the salt cellar", "the reading room", "the breakwater", "the old chapel",
    "the fish market", "the signal tower",
]
_OBJECTS = [
    "a brass key", "the tide ledger", "a coil of rope", "the storm glass",
    "a jar of ink", "the harbour chart", "a broken lantern", "the log book",
    "a bundle of letters", "the small telescope",
]
_WEATHER = [
    "fog rolled in from the sea", "the rain kept its slow count",
    "gulls argued over the masts", "the wind turned to the east",
    "frost silvered the railings", "the tide slipped out early",
    "thunder walked along the horizon", "the morning stayed grey and still",
]
_TEMPLATES = [
    "{name} carried {obj} down to {place} before the bells rang.",
    "At {place}, {name} wrote another line in the ledger and waited.",
    "Nobody remembered why {obj} was kept at {place}, least of all {name}.",
    "When {weather}, {name} locked {place} and walked home along the wall.",
    "{name} said the sea keeps its own ledger, and {place} keeps ours.",
    "Every spring {name} counted the boats twice and misplaced {obj} once.",
    "The letters in {obj} were addressed to {place}, in a hand like knotted rope.",
    "{name} traded {obj} for a story about {place} that was mostly true.",
    "By dusk {weather}, and the lamps of {place} came on one by one.",
    "{name} kept {obj} wrapped in oilcloth beneath the stairs of {place}.",
    "It was said that {place} answered questions if you asked them slowly.",
    "{name} taught the children to read the tide from {place}, patiently.",
    "In the drawer of {place} lay {obj}, and under it, a second drawer.",
    "{name} never crossed to {place} on a Sunday, and never said why.",
    "The ledger noted only this: {weather}, and {obj} was returned.",
    "Half the town swore {name} could hear the fog before it arrived.",
    "{name} repaired {obj} with wire and patience borrowed from {place}.",
    "A visitor once asked the way to {place}; {name} pointed at the sea.",
    "All winter {weather}, and still {name} opened {place} at seven.",
    "The catalogue of {place} listed {obj} twice, which pleased {name}.",
]


def build_corpus(target_bytes: int = 50_000, seed: int = 1) -> str:
    """Deterministically expand the hand-written templates to ~target_bytes."""
    rng = random.Random(seed)
    parts: list[str] = []
    size = 0
    while size < target_bytes:
        n_sent = rng.randint(4, 7)
        sents = []
        for _ in range(n_sent):
            t = rng.choice(_TEMPLATES)
            s = t.format(
                name=rng.choice(_NAMES),
                place=rng.choice(_PLACES),
                obj=rng.choice(_OBJECTS),
                weather=rng.choice(_WEATHER),
            )
            sents.append(s)
        para = " ".join(sents) + "\n\n"
        parts.append(para)
        size += len(para)
    return "".join(parts)[:target_bytes]


def build_alt_corpus(target_bytes: int = 50_000, seed: int = 2) -> str:
    """A stylistically different corpus: terse pseudo-code and arithmetic."""
    rng = random.Random(seed)
    ops = ["+", "-", "*"]
    parts: list[str] = []
    size = 0
    while size < target_bytes:
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        v1, v2, v3 = (rng.randint(0, 9) for _ in range(3))
        op = rng.choice(ops)
        line = (
            f"let x{v1} = (v{v2} {op} v{v3}) * {a};\n"
            f"if x{v1} > {b} then emit(x{v1}) else skip;\n"
        )
        parts.append(line)
        size += len(line)
    return "".join(parts)[:target_bytes]


class Vocab:
    """Char-level vocabulary over one or more corpora (shared token space)."""

    def __init__(self, *texts: str):
        chars = sorted(set("".join(texts)))
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        """Map a string to a list of token ids."""
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        """Map token ids back to a string."""
        return "".join(self.itos[int(i)] for i in ids)
