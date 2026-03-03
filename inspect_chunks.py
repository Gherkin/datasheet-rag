"""Quick inspection of the abstractive chunk graph output."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "output/8222e0f32bc0a22ca63d2241c148249327384cd2f87f0fe0bc2e95cc3d9b6e26_chunks_abstractive.json"

data = json.load(open(path))
chunks = data["chunks"]

by_level: dict[int, list] = {0: [], 1: [], 2: []}
for c in chunks.values():
    by_level[c["level"]].append(c)

print(f"MACRO(0): {len(by_level[0])}  MESO(1): {len(by_level[1])}  MICRO(2): {len(by_level[2])}")
print()

# ── MACRO summaries (what Claude produced) ──────────────────────────────────
print("=" * 70)
print("MACRO CHAPTER SUMMARIES  (Claude Haiku abstractive output)")
print("=" * 70)
for m in by_level[0]:
    meta = m["metadata"]
    title = meta.get("chapter_title") or "(untitled)"
    pages = meta.get("page_numbers", [])
    children = len(m.get("children_ids") or [])
    tokens = m.get("token_count", 0)
    print(f"\nChapter : {title}")
    print(f"Pages   : {pages}  |  MESO children: {children}  |  Tokens: {tokens}")
    print(f"Summary :")
    text = m["text"]
    # Print up to 600 chars, word-wrapped at ~80 cols
    for i in range(0, min(len(text), 600), 80):
        print(f"  {text[i:i+80]}")
    if len(text) > 600:
        print("  [... truncated]")

# ── Sample MESO ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAMPLE MESO CHUNK (index 5)")
print("=" * 70)
meso = by_level[1][5]
print(f"Section : {meso['metadata'].get('section_title')}")
print(f"Context : {meso['metadata'].get('context_string')}")
print(f"Pages   : {meso['metadata'].get('page_numbers')}")
print(f"Tokens  : {meso.get('token_count')}")
print(f"Prev    : {meso.get('prev_id')}")
print(f"Next    : {meso.get('next_id')}")
print(f"Text    :\n  {meso['text'][:400]}")

# ── Sample MICRO ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAMPLE MICRO CHUNK (index 10)")
print("=" * 70)
micro = by_level[2][10]
print(f"Context : {micro['metadata'].get('context_string')}")
print(f"Pages   : {micro['metadata'].get('page_numbers')}")
print(f"Tokens  : {micro.get('token_count')}")
print(f"Parent  : {micro.get('parent_id')}")
print(f"Text    :\n  {micro['text'][:400]}")

# ── Figure chunks ────────────────────────────────────────────────────────────
figures = [c for c in chunks.values() if c.get("figure_image_path")]
print(f"\n{'=' * 70}")
print(f"FIGURE CHUNKS: {len(figures)}")
print("=" * 70)
for f in figures[:3]:
    print(f"  Path    : {f['figure_image_path']}")
    print(f"  Caption : {f.get('figure_caption')}")
    print(f"  Context : {f['metadata'].get('context_string')}")
    print()
