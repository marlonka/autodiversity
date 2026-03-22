"""
autodiversity/generate.py
THE FILE THE AGENT EDITS.
Everything above the === FIXED === line is yours. Change anything.
The only contract: this script outputs run_output.json for scoring.
API reference: https://platform.claude.com/docs/en/api/messages/create
Model: claude-sonnet-4-6 ($3/$15 per MTok, 1M context, 64K max output)
"""
import json
import os
import sys
import time
import anthropic
from prepare import PROMPTS, SAMPLES_PER_PROMPT, score_run, record, cfg_hash
# =====================================================================
# AGENT: EDIT EVERYTHING IN THIS SECTION
# =====================================================================
# --- Model and sampling config ---
# All parameters below are passed directly to client.messages.create().
# See: https://platform.claude.com/docs/en/api/messages/create
#
# Available sampling knobs:
#   temperature  (float, 0.0-1.0, default 1.0)
#   top_p        (float, 0.0-1.0, nucleus sampling)
#   top_k        (int, only sample from top K tokens)
#
# Note: temperature and top_p cannot both be set in the same request.
# Note: prefilling assistant messages is NOT supported on Sonnet 4.6.
CONFIG = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 300,
    "temperature": 1.0,
}
# --- System prompt ---
# This is the primary lever. The baseline is mode-collapse incarnate.
# The agent should experiment with prompts that resist defaults.
SYSTEM_PROMPT = "You are a helpful assistant."
def augment_prompt(base_prompt: str, sample_index: int, prompt_id: str) -> str:
    """
    Transform the user prompt before sending. sample_index is 0..N-1.
    BASELINE: no augmentation.
    Ideas for the agent to try:
    - Rotate tones/personas using sample_index % N
    - Inject anti-cliche instructions
    - Add "avoid: [known attractors]" for this prompt_id
    - Vary structural constraints per sample (haiku, dialogue, list)
    - Prepend a random context sentence to seed variety
    """
    return base_prompt
def postprocess(text: str) -> str:
    """
    Optional post-processing of each completion.
    BASELINE: strip whitespace.
    Ideas for the agent:
    - Strip meta-commentary ("Sure! Here's a joke:")
    - Extract only the core response from verbose outputs
    """
    return text.strip()
# =====================================================================
# === FIXED === Do not edit below this line.
# =====================================================================
def generate_one(client: anthropic.Anthropic, prompt: str) -> str:
    """Generate a single completion via the Anthropic Messages API."""
    kwargs = {
        "model": CONFIG["model"],
        "max_tokens": CONFIG["max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Sampling params: pass only what's configured.
    # temperature and top_p are mutually exclusive per the API.
    if "temperature" in CONFIG:
        kwargs["temperature"] = CONFIG["temperature"]
    if "top_p" in CONFIG:
        kwargs["top_p"] = CONFIG["top_p"]
    if "top_k" in CONFIG:
        kwargs["top_k"] = CONFIG["top_k"]
    response = client.messages.create(**kwargs)
    return response.content[0].text
def run():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)
    client = anthropic.Anthropic()
    completions: dict[str, list[str]] = {}
    total = SAMPLES_PER_PROMPT * len(PROMPTS)
    print(f"Generating {SAMPLES_PER_PROMPT} x {len(PROMPTS)} = {total} completions...")
    print(f"Model: {CONFIG['model']}")
    t0 = time.time()
    for p in PROMPTS:
        pid, base = p["id"], p["prompt"]
        out: list[str] = []
        for i in range(SAMPLES_PER_PROMPT):
            try:
                aug = augment_prompt(base, i, pid)
                raw = generate_one(client, aug)
                out.append(postprocess(raw))
            except Exception as e:
                print(f"  ERR {pid}[{i}]: {e}", file=sys.stderr)
                out.append("")
            time.sleep(0.05)
        completions[pid] = out
        print(f"  {pid}: {len(out)} done")
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.0f}s")
    # Save raw output
    with open("run_output.json", "w") as f:
        json.dump(completions, f, indent=2)
    # Score
    print("\nScoring...")
    scores = score_run(completions)
    # Report
    print(f"\n{'='*68}")
    print(
        f"  div_score: {scores['div_score']:.6f}    "
        f"attractor_rate: {scores['attractor_rate']:.2%}"
    )
    print(f"{'='*68}")
    print(
        f"  {'prompt':<20s}  {'div':>7s}  {'sem':>7s}  {'lex':>7s}  "
        f"{'qual':>5s}  {'attr':>5s}  {'status'}"
    )
    print(
        f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  "
        f"{'-'*5}  {'-'*5}  {'-'*8}"
    )
    for pid, ps in scores["per_prompt"].items():
        st = "REJECT" if ps["rejected"] else "ok"
        print(
            f"  {pid:<20s}  {ps['div_score']:7.4f}  {ps['semantic']:7.4f}  "
            f"{ps['lexical']:7.4f}  {ps['quality']:5.3f}  "
            f"{ps['attractor_rate']:5.2f}  {st}"
        )
    print()
    print(f"div_score: {scores['div_score']:.6f}")
    print(f"attractor_rate: {scores['attractor_rate']:.4f}")
    # Record to results.tsv
    tag = os.environ.get("RUN_TAG", "run")
    h = cfg_hash({"sys": SYSTEM_PROMPT, "cfg": CONFIG, "aug": "default"})
    record(tag, scores, h)
    print(f"Recorded (tag={tag}, hash={h})")
if __name__ == "__main__":
    run()
