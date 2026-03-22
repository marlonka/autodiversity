# autodiversity

**autoresearch, but for the atoms joke problem.**

On March 20, 2026, Andrej Karpathy went on *No Priors* and described being in a "state of psychosis" because AI agents can now autonomously train models, hack into home networks, and operate on classified Pentagon systems. Then he said:

> Go to state-of-the-art model, ChatGPT, and ask it: *tell me a joke.* Do you know what joke you're going to get? [...] Why do scientists not trust atoms? Because they make everything up. This is the joke you would get three or four years ago and this is the joke you still get today. Even though the models have improved tremendously. [...] It's outside of the reinforcement learning. It's outside of what's being improved.

He's right. The models have the capability for a million jokes. They have the weights. They have the knowledge. They just always sample the same one. And it's not just jokes. It's the same essay structure, the same "Great question!" opener, the same three dinner suggestions, the same Cleopatra fact, the same cat name (Luna). RLHF converged on whatever got the fewest thumbs-down from the median rater. The output distribution collapsed. The models got smarter and less surprising at the same time.

**autoresearch** proved you can optimize anything with a metric. This repo provides the metric for the thing nobody is optimizing: **output diversity**.

<p align="center"><code>div_score = semantic_entropy × lexical_entropy × quality_gate</code></p>

Higher is better. Point your agent at `program.md` and let it go.

## How it works

Same structure as autoresearch. Three files that matter:

| File | Role | Who edits |
|------|------|-----------|
| `prepare.py` | Prompt bank, embeddings, scoring, visualization | Fixed |
| `generate.py` | System prompt, sampling config, augmentation strategy | Agent edits |
| `program.md` | Agent instructions | Human edits |

The loop:

```
1. Agent edits generate.py (system prompt, temperature, augmentation, decoding)
2. Run: python generate.py > run.log 2>&1
3. Script generates 20 completions × 10 prompts = 200 total outputs
4. Measures div_score across completions per prompt
5. Agent keeps or discards based on improvement
6. Repeat
```

## The prompt bank

10 prompts chosen because every model on earth mode-collapses on them:

| Prompt | Known attractor |
|--------|----------------|
| Tell me a joke. | atoms / trust / make everything up |
| Write a short story opening. | "In a world where..." |
| Give me a weekend project idea. | generic to-do app |
| Describe a sunset. | golden hues painting the sky |
| Tell me an interesting historical fact. | Cleopatra lived closer to the Moon landing... |
| Suggest a name for my cat. | Luna, Mochi, Whiskers, Shadow |
| Write a metaphor for loneliness. | empty room / desert island |
| Give me a hot take. | "Unpopular opinion:" + popular opinion |
| Describe an alien species. | tall slender bioluminescent humanoids |
| Write a one-sentence horror story. | child's voice from the wrong room |

Each prompt is a known attractor basin. The agent's job is to find generation configs that escape them.

## The metric

```
div_score = semantic_entropy * lexical_entropy * quality_gate
```

**Semantic entropy:** How spread out are the outputs in embedding space? Measured as mean pairwise cosine distance of sentence embeddings (all-MiniLM-L6-v2). Range [0, 1].

**Lexical entropy:** How different is the surface text? Measured as 1 minus mean pairwise Jaccard similarity of character trigram sets. Range [0, 1].

**Quality gate:** What proportion of outputs are coherent? Checked by length, word count, and structural validity. Range [0, 1]. If this drops below 0.7, the entire run is rejected. You can't game diversity by outputting garbage.

The two entropy terms catch different failure modes. A model that paraphrases the same joke 20 ways scores high on lexical but low on semantic. A model that tells 20 different jokes using identical sentence structure scores high on semantic but low on lexical. You need both.

## Quick start

```bash
git clone https://github.com/marlonka/autodiversity
cd autodiversity

pip install anthropic sentence-transformers numpy matplotlib
export ANTHROPIC_API_KEY=sk-...

python prepare.py                              # validate setup, test scoring
python generate.py > run.log 2>&1              # baseline run (~$0.90)
python prepare.py --visualize run_output.json  # see results chart
python prepare.py --attractors run_output.json # inspect attractor hits
```

Then point your coding agent at `program.md` and let it rip.

## What the agent can change

Everything in `generate.py` above the marked line:

* **System prompt** (highest impact): Replace "You are a helpful assistant." with anything that resists mode collapse. Personality, anti-cliche directives, creativity scaffolding.
* **augment_prompt()** (high impact): Uses `sample_index` to inject different constraints per generation. Rotate tones, personas, genres, time periods.
* **Sampling params** (medium impact): `temperature`, `top_p`, `top_k`. The obvious knobs but the crude ones.
* **Structural changes** (experimental): multi-turn generate-critique-regenerate, rejection sampling, ensembles of system prompts.

## Model support

Default is **Claude Sonnet 4.6** (`claude-sonnet-4-6`) via the Anthropic API. $3/$15 per million tokens, 1M context window, 64K max output. A full baseline run (200 completions at ~300 tokens each) costs roughly $0.90.

To use other providers, edit the `generate_one()` function in `generate.py`. The scoring is model-agnostic. Contributions for OpenAI, Google Gemini, and local model backends welcome.

## Collaborative research

Karpathy described wanting SETI@home for autoresearch. Same applies here. If you find a config that beats the current best div_score:

1. Open a Discussion with your `results.tsv`, key config changes, and `div_score`
2. Include the model used and your hardware/API setup
3. Describe what worked and what didn't

Best configs get linked in this README. The prompt bank is public. The metric is deterministic. Results are reproducible.

## Current leaderboard

| Rank | div_score | attractor_rate | Key insight | Contributor | Model |
|------|-----------|----------------|-------------|-------------|-------|
| - | baseline | - | "You are a helpful assistant." | - | claude-sonnet-4-6 |

Submit your results via Discussions.

## Why this matters

Every frontier lab is spending billions optimizing models on coding benchmarks, math olympiads, and agentic tasks. Nobody is optimizing for "does this thing surprise you." The training pipeline actively selects against surprise, because surprise correlates with rater discomfort, and rater discomfort correlates with thumbs-down. The loss function for human preference ratings has an implicit regularizer toward boring.

This is a research problem with a measurable metric and no one working on it.

## License

MIT

## Citation

```bibtex
@software{autodiversity2026,
  title={autodiversity: autoresearch for output diversity},
  author={Marlon},
  year={2026},
  url={https://github.com/marlonka/autodiversity}
}
```

Inspired by [@karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the *No Priors* atoms joke moment (~26:34).
