"""Build LibriSpeech manifest: flac + transcript -> ARPABET phone sequence.

Runs under the 3.11 env (needs g2p.py / espeak-ng). Output: gop_data_libri/manifest.json
  {utt_id: {split, wav, text, phones, phone_ids}}
Utts with adjacent-duplicate phones (CTC-unspeakable) or unknown phones are dropped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from g2p import text_to_arpabet
from gop_data import (
    PROJECT_ROOT,
    build_phone_inventory,
    load_scores,
    phone_to_id_map,
)
import g2p as g2p_mod

LIBRI_ROOT = PROJECT_ROOT / "wavs" / "LibriSpeech"
OUT_DIR = PROJECT_ROOT / "gop_data_libri"


def iter_libri_utts(split: str):
    root = LIBRI_ROOT / split
    for spk in sorted(p for p in root.iterdir() if p.is_dir()):
        for chap in sorted(p for p in spk.iterdir() if p.is_dir()):
            trans = chap / f"{spk.name}-{chap.name}.trans.txt"
            lines: dict[str, str] = {}
            if trans.is_file():
                for line in trans.read_text(encoding="utf-8").splitlines():
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        lines[parts[0]] = parts[1].strip()
            for flac in sorted(chap.glob("*.flac")):
                utt = flac.stem
                text = lines.get(utt)
                if text:
                    yield utt, str(flac), text


def main() -> None:
    inventory = build_phone_inventory(load_scores())
    inv_map = phone_to_id_map(inventory)
    print(f"phone inventory ({len(inventory)}): {inventory}")

    manifest: dict[str, dict] = {}
    skipped_unknown = 0
    n_collapsed = 0
    t0 = time.perf_counter()
    for split in ("train-clean-100", "test-clean"):
        n_split = 0
        for utt, wav, text in iter_libri_utts(split):
            phones = text_to_arpabet(text)
            ids = [inv_map[p] for p in phones if p in inv_map]
            if len(ids) != len(phones):
                skipped_unknown += 1
                continue
            collapsed_phones, collapsed_ids = [], []
            for p, i in zip(phones, ids):
                if not collapsed_ids or i != collapsed_ids[-1]:
                    collapsed_phones.append(p)
                    collapsed_ids.append(i)
            n_collapsed += len(ids) - len(collapsed_ids)
            manifest[utt] = {
                "split": split, "wav": wav, "text": text,
                "phones": collapsed_phones, "phone_ids": collapsed_ids,
            }
            n_split += 1
        print(f"  {split}: {n_split} utts kept ({time.perf_counter()-t0:.0f}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"done: {len(manifest)} utts, {skipped_unknown} skipped (unknown phones), "
        f"{n_collapsed} adjacent-dup phones collapsed, {time.perf_counter()-t0:.0f}s"
    )
    if g2p_mod._seen_unknown:
        print("unknown IPA chars seen:", sorted(g2p_mod._seen_unknown))


if __name__ == "__main__":
    main()
