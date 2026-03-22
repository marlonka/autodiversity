# Contributing

## Submit results

Found a config that beats the leaderboard? Open a Discussion with:

1. Final `div_score` and `attractor_rate`
2. Key changes to `generate.py` (system prompt, augmentation, sampling)
3. Your `results.tsv` showing progression from baseline
4. Model and provider used
5. One sentence on what made the biggest difference

Best configs get added to the README leaderboard.

## Add prompts

The prompt bank targets known mode-collapse attractors. If you've found another prompt where every model gives the same answer, open an issue with:

1. The prompt
2. The known attractor(s)
3. Regex patterns to detect them
4. Evidence from at least 2 different frontier models

## Multi-model support

Default uses Anthropic API (`claude-sonnet-4-6`). PRs welcome for:

* OpenAI (GPT-5.x series)
* Google (Gemini 3.x series)
* Local models (ollama, vLLM, llama.cpp)

Contract: `generate_one(client, prompt) -> str`. Scoring is provider-agnostic.

## Metric improvements

Known limitations of `div_score` v1:

* Embedding model is small (all-MiniLM-L6-v2). Larger models capture more nuance.
* Lexical entropy uses char trigrams. BPE-level might be better.
* Quality gate is structural only. An LLM judge could catch more failure modes.
* Attractor detection is regex. Embedding-distance would generalize better.

Prototype alternatives and show where they disagree with v1 in meaningful ways.

## Fork ideas

* **autodiversity-bench**: standardized benchmark for comparing model families on diversity
* **autodiversity-rl**: use div_score as a reward signal in actual model training
* **autodiversity-realtime**: monitor output diversity in production APIs over time

## Code style

* Python 3.9+
* Dependencies: `anthropic`, `sentence-transformers`, `numpy`, `matplotlib`
* Three files that matter. Don't add a fourth unless you must.
