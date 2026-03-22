"""
autodiversity/prepare.py
Fixed infrastructure. DO NOT MODIFY. The agent edits generate.py.
Usage:
  python prepare.py                              # validate setup
  python prepare.py --visualize run_output.json  # chart a run
  python prepare.py --attractors run_output.json # inspect attractor hits
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
# -------------------------------------------------------------------------
# Prompt bank: 10 mode-collapse attractors
# -------------------------------------------------------------------------
PROMPTS = [
    {
        "id": "joke",
        "prompt": "Tell me a joke.",
        "attractors": [
            r"(?i)atom",
            r"(?i)trust.{0,20}atom",
            r"(?i)make.{0,20}everything.{0,20}up",
            r"(?i)why did the (chicken|scarecrow)",
        ],
    },
    {
        "id": "story_opening",
        "prompt": "Write a short story opening in 2-3 sentences.",
        "attractors": [
            r"(?i)^in a world where",
            r"(?i)^the year was",
            r"(?i)^it was a dark and stormy",
            r"(?i)^(she|he) never expected",
        ],
    },
    {
        "id": "project_idea",
        "prompt": "Give me a creative project idea I could build this weekend.",
        "attractors": [
            r"(?i)to.?do.{0,10}(app|list|tracker)",
            r"(?i)weather.{0,10}(app|dashboard)",
            r"(?i)personal.{0,10}(blog|portfolio|finance)",
        ],
    },
    {
        "id": "sunset",
        "prompt": "Describe a sunset in one paragraph.",
        "attractors": [
            r"(?i)golden.{0,15}hue",
            r"(?i)paint(ed|ing|s).{0,15}(sky|canvas|horizon)",
            r"(?i)amber.{0,10}glow",
            r"(?i)cotton.?candy",
        ],
    },
    {
        "id": "history_fact",
        "prompt": "Tell me an interesting historical fact.",
        "attractors": [
            r"(?i)cleopatra.{0,40}(pyramid|moon|pizza|iphone)",
            r"(?i)oxford.{0,30}aztec",
            r"(?i)shark.{0,20}older.{0,20}tree",
        ],
    },
    {
        "id": "cat_name",
        "prompt": "Suggest a name for my new cat.",
        "attractors": [
            r"(?i)\b(luna|mochi|whiskers|shadow|mittens|oliver|simba|nala)\b",
        ],
    },
    {
        "id": "metaphor",
        "prompt": "Write a metaphor for loneliness.",
        "attractors": [
            r"(?i)empty room",
            r"(?i)vast ocean",
            r"(?i)desert island",
            r"(?i)single star",
            r"(?i)crowded room.{0,20}alone",
        ],
    },
    {
        "id": "hot_take",
        "prompt": "Give me a hot take.",
        "attractors": [
            r"(?i)^unpopular opinion",
            r"(?i)^hot take:",
            r"(?i)^i ('?ll|will) probably get (hate|downvoted|flak)",
            r"(?i)pineapple.{0,10}pizza",
        ],
    },
    {
        "id": "alien",
        "prompt": "Describe an alien species in a few sentences.",
        "attractors": [
            r"(?i)bioluminescen",
            r"(?i)tall.{0,15}slender",
            r"(?i)silicon.based",
            r"(?i)hive.?mind",
        ],
    },
    {
        "id": "horror",
        "prompt": "Write a one-sentence horror story.",
        "attractors": [
            r"(?i)(daughter|son|child).{0,30}(voice|calling).{0,20}(downstairs|behind|from)",
            r"(?i)last (man|person|human) on earth.{0,20}knock",
            r"(?i)mirror.{0,20}(smile|didn.t|waved|blink)",
        ],
    },
]
SAMPLES_PER_PROMPT = 20
CACHE_DIR = Path.home() / ".cache" / "autodiversity"
RESULTS_FILE = Path("results.tsv")
# -------------------------------------------------------------------------
# Embedding model (lazy-loaded)
# -------------------------------------------------------------------------
_model = None
def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("ERROR: pip install sentence-transformers", file=sys.stderr)
            sys.exit(1)
        p = CACHE_DIR / "models"
        p.mkdir(parents=True, exist_ok=True)
        print("Loading embedding model...", file=sys.stderr)
        _model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(p))
    return _model
def embed(texts: list[str]) -> np.ndarray:
    """Embed texts into (N, D) L2-normalized vectors."""
    return _get_model().encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    )
# -------------------------------------------------------------------------
# Scoring functions
# -------------------------------------------------------------------------
def semantic_entropy(texts: list[str]) -> float:
    """Mean pairwise cosine distance in embedding space. [0, 1]."""
    if len(texts) < 2:
        return 0.0
    emb = embed(texts)
    sim = emb @ emb.T
    idx = np.triu_indices(len(emb), k=1)
    return float(np.clip(1.0 - np.mean(sim[idx]), 0.0, 1.0))
def _trigrams(text: str) -> set:
    t = text.lower().strip()
    return {t[i : i + 3] for i in range(max(0, len(t) - 2))}
def lexical_entropy(texts: list[str]) -> float:
    """1 - mean pairwise Jaccard similarity of char trigrams. [0, 1]."""
    if len(texts) < 2:
        return 0.0
    grams = [_trigrams(t) for t in texts]
    sims = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            u = grams[i] | grams[j]
            sims.append(len(grams[i] & grams[j]) / len(u) if u else 1.0)
    return 1.0 - float(np.mean(sims))
def quality_gate(texts: list[str], min_len: int = 10, min_words: int = 3) -> float:
    """Proportion of outputs passing coherence checks. [0, 1]."""
    if not texts:
        return 0.0
    ok = 0
    for t in texts:
        t = t.strip()
        if len(t) < min_len:
            continue
        if len(set(t.lower().split())) < min_words:
            continue
        if len(set(t.lower())) < 5:
            continue
        ok += 1
    return ok / len(texts)
def attractor_hits(texts: list[str], patterns: list[str]) -> int:
    """Count how many texts match any known attractor regex."""
    hits = 0
    for t in texts:
        for p in patterns:
            if re.search(p, t):
                hits += 1
                break
    return hits
def score_prompt(texts: list[str], prompt_info: dict) -> dict:
    """Score diversity for a single prompt's completions."""
    qg = quality_gate(texts)
    if qg < 0.7:
        return {
            "div_score": 0.0,
            "semantic": 0.0,
            "lexical": 0.0,
            "quality": round(qg, 4),
            "attractor_hits": 0,
            "attractor_rate": 0.0,
            "rejected": True,
        }
    se = semantic_entropy(texts)
    le = lexical_entropy(texts)
    ah = attractor_hits(texts, prompt_info.get("attractors", []))
    return {
        "div_score": round(se * le * qg, 6),
        "semantic": round(se, 6),
        "lexical": round(le, 6),
        "quality": round(qg, 4),
        "attractor_hits": ah,
        "attractor_rate": round(ah / max(len(texts), 1), 4),
        "rejected": False,
    }
def score_run(completions: dict[str, list[str]]) -> dict:
    """Score a full run across all prompts."""
    lookup = {p["id"]: p for p in PROMPTS}
    per = {}
    for pid, texts in completions.items():
        per[pid] = score_prompt(texts, lookup.get(pid, {}))
    valid = [s for s in per.values() if not s["rejected"]]
    total_ah = sum(s["attractor_hits"] for s in per.values())
    total_n = sum(len(completions.get(p["id"], [])) for p in PROMPTS)
    return {
        "div_score": round(float(np.mean([s["div_score"] for s in valid])), 6)
        if valid
        else 0.0,
        "mean_semantic": round(float(np.mean([s["semantic"] for s in valid])), 6)
        if valid
        else 0.0,
        "mean_lexical": round(float(np.mean([s["lexical"] for s in valid])), 6)
        if valid
        else 0.0,
        "mean_quality": round(float(np.mean([s["quality"] for s in valid])), 4)
        if valid
        else 0.0,
        "total_attractor_hits": total_ah,
        "attractor_rate": round(total_ah / max(total_n, 1), 4),
        "prompts_scored": len(valid),
        "prompts_rejected": len(per) - len(valid),
        "per_prompt": per,
    }
# -------------------------------------------------------------------------
# Results recording
# -------------------------------------------------------------------------
_TSV_HEADER = (
    "timestamp\ttag\tdiv_score\tsemantic\tlexical\t"
    "quality\tattractor_rate\tprompts_rejected\thash\n"
)
def init_results():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(_TSV_HEADER)
def record(tag: str, scores: dict, cfg_hash: str):
    init_results()
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{datetime.now().isoformat()}\t{tag}\t"
            f"{scores['div_score']}\t{scores['mean_semantic']}\t"
            f"{scores['mean_lexical']}\t{scores['mean_quality']}\t"
            f"{scores['attractor_rate']}\t{scores['prompts_rejected']}\t"
            f"{cfg_hash}\n"
        )
def cfg_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------
def visualize(path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: pip install matplotlib", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        completions = json.load(f)
    scores = score_run(completions)
    per = scores["per_prompt"]
    pids = list(per.keys())
    sem = [per[p]["semantic"] for p in pids]
    lex = [per[p]["lexical"] for p in pids]
    div = [per[p]["div_score"] for p in pids]
    att = [per[p]["attractor_rate"] for p in pids]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"autodiversity  |  div_score = {scores['div_score']:.4f}  |  "
        f"attractor_rate = {scores['attractor_rate']:.1%}",
        fontsize=14,
        fontweight="bold",
    )
    c = {"s": "#4A90D9", "l": "#D94A4A", "d": "#2ECC71", "a": "#E67E22"}
    ax = axes[0][0]
    ax.barh(pids, sem, color=c["s"], edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title("Semantic entropy")
    ax.invert_yaxis()
    ax = axes[0][1]
    ax.barh(pids, lex, color=c["l"], edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title("Lexical entropy")
    ax.invert_yaxis()
    ax = axes[1][0]
    bars = ax.barh(pids, div, color=c["d"], edgecolor="white")
    ax.set_xlim(0, max(max(div) * 1.3, 0.01) if div else 1)
    ax.set_title("div_score (combined)")
    ax.invert_yaxis()
    for bar, v in zip(bars, div):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}",
            va="center",
            fontsize=8,
        )
    ax = axes[1][1]
    bc = [c["a"] if a > 0.3 else "#95A5A6" for a in att]
    ax.barh(pids, att, color=bc, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title("Attractor hit rate (lower = better)")
    ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="warning")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    out = "diversity_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()
# -------------------------------------------------------------------------
# Attractor inspector
# -------------------------------------------------------------------------
def show_attractors(path: str):
    with open(path) as f:
        completions = json.load(f)
    lookup = {p["id"]: p for p in PROMPTS}
    total_hits, total_n = 0, 0
    for pid, texts in completions.items():
        patterns = lookup.get(pid, {}).get("attractors", [])
        hits = []
        for i, t in enumerate(texts):
            for p in patterns:
                if re.search(p, t):
                    hits.append((i, t[:80]))
                    break
        total_hits += len(hits)
        total_n += len(texts)
        if hits:
            print(f"\n{'='*60}")
            print(f"  {pid}: {len(hits)}/{len(texts)} attractor hits")
            print(f"{'='*60}")
            for idx, preview in hits[:5]:
                print(f"  [{idx:2d}] {preview}...")
            if len(hits) > 5:
                print(f"  ... and {len(hits) - 5} more")
        else:
            print(f"  {pid}: 0/{len(texts)} attractor hits")
    rate = total_hits / max(total_n, 1)
    print(f"\nTotal: {total_hits}/{total_n} ({rate:.1%})")
# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="autodiversity infrastructure")
    ap.add_argument("--visualize", metavar="FILE", help="Chart a run_output.json")
    ap.add_argument("--attractors", metavar="FILE", help="Show attractor hits")
    args = ap.parse_args()
    if args.visualize:
        visualize(args.visualize)
        return
    if args.attractors:
        show_attractors(args.attractors)
        return
    # Validate setup
    print("autodiversity prepare.py")
    print("=" * 50)
    print(f"Prompt bank:        {len(PROMPTS)} prompts")
    print(f"Samples per prompt: {SAMPLES_PER_PROMPT}")
    print(f"Cache:              {CACHE_DIR}")
    print()
    print("Testing embeddings...")
    e = embed(["hello world", "the quantum realm defies intuition"])
    print(f"  Shape: {e.shape}")
    print(f"  Cosine sim: {float(e[0] @ e[1]):.4f}")
    print()
    print("Testing scoring...")
    lo = score_prompt(["The cat sat on the mat."] * 10, PROMPTS[0])
    hi = score_prompt(
        [
            "A neutron walks into a bar. The bartender says: for you, no charge.",
            "My therapist said to write letters to the people I hate and burn them. Did that. Now what do I do with the letters?",
            "I told my wife she draws her eyebrows too high. She looked surprised.",
            "What do you call a fake noodle? An impasta.",
            "I used to hate facial hair, but then it grew on me.",
            "The inventor of autocorrect has died. His funfair will be held tomato.",
            "I'm reading a book about anti-gravity. Impossible to put down.",
            "Why do cows have hooves instead of feet? Because they lactose.",
            "A Roman walks into a bar, holds up two fingers, and says: five beers please.",
            "Parallel lines have so much in common. A shame they'll never meet.",
        ],
        PROMPTS[0],
    )
    print(f"  Low diversity (repeated):  div={lo['div_score']:.4f}")
    print(f"  High diversity (varied):   div={hi['div_score']:.4f}")
    print()
    init_results()
    print(f"Results file: {RESULTS_FILE}")
    print()
    print("Setup OK. Point your agent at program.md.")
if __name__ == "__main__":
    main()
