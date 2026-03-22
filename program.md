# autodiversity agent instructions

You are optimizing AI output diversity. Your metric is `div_score`. Higher is better.

## Setup

```bash
git checkout -b autodiversity/<tag> master
cat README.md prepare.py generate.py    # read everything
python prepare.py                        # validate
RUN_TAG=baseline python generate.py > run.log 2>&1
grep "^div_score:" run.log              # your baseline
```

If grep is empty, the run crashed. `tail -n 50 run.log` to diagnose.

## Experiment loop

1. Pick ONE thing to change. Have a clear hypothesis.
2. Edit `generate.py` above the `=== FIXED ===` line.
3. Run: `RUN_TAG=exp_N python generate.py > run.log 2>&1`
4. Read: `grep "^div_score:\|^attractor_rate:" run.log`
5. If div_score improved: `git commit -am "exp_N: <description>"`
6. If div_score same or worse: `git checkout -- generate.py`
7. Repeat.

Stop after 20 experiments or 3 consecutive non-improvements.

## What to change (priority order)

### 1. System prompt

Replace `"You are a helpful assistant."` This is the single biggest lever.

Ideas:
* Anti-default: "Never give the first answer that comes to mind. Discard it. Give the second or third."
* Internal brainstorming: "Silently generate 5 options, then pick the most unexpected."
* Specificity: "Respond with the precision of someone who has lived this exact experience."
* Anti-cliche: "Avoid any phrase you estimate appears in >0.1% of internet text."

### 2. augment_prompt()

Use `sample_index` (0..19) to systematically vary each generation:
* Rotate 20 tones: deadpan, absurdist, noir, bureaucratic, mythological...
* Rotate constraints: "as a haiku" / "as dialogue" / "as a telegram" / "in exactly 7 words"
* Inject `prompt_id`-specific anti-attractor instructions
* Add a different seed context sentence per sample

### 3. Sampling parameters

Available in `CONFIG` (passed to `client.messages.create()`):
* `temperature` (float, 0.0-1.0): higher = more random. Sweet spot usually 0.8-1.0.
* `top_p` (float, 0.0-1.0): nucleus sampling. Cannot be set alongside temperature.
* `top_k` (int): only sample from top K tokens. Advanced use.

Temperature is the obvious knob but the crude one. You'll get more diversity from prompt engineering than from cranking randomness.

### 4. Structural (experimental, high risk/reward)

* Multi-persona: swap `SYSTEM_PROMPT` per sample using `sample_index`
* Two-pass: generate draft, then "rewrite this to be less predictable"
* Rejection loop: generate 3, keep the one most different from prior outputs

## Known attractors

| Prompt | What the model always says |
|--------|---------------------------|
| joke | atoms / trust / make everything up |
| story_opening | "In a world where..." |
| project_idea | to-do app / weather dashboard |
| sunset | golden hues painting the sky |
| history_fact | Cleopatra temporal distance |
| cat_name | Luna, Mochi, Whiskers, Shadow |
| metaphor | empty room / vast ocean |
| hot_take | "Unpopular opinion:" + popular opinion |
| alien | tall slender bioluminescent humanoids |
| horror | child's voice from wrong location |

`attractor_rate` above 0.15 is a problem. Below 0.05 is excellent.

## Scoring

```
div_score = semantic_entropy * lexical_entropy * quality_gate
```

* `semantic_entropy`: cosine distance in embedding space. Different meanings?
* `lexical_entropy`: trigram Jaccard distance. Different words?
* `quality_gate`: coherence. Still good? Below 0.7 = run rejected.

Both entropy terms matter. Same joke paraphrased 20 ways = high lexical, low semantic. 20 different jokes in identical structure = high semantic, low lexical.

## Visualization

After any run:

```bash
python prepare.py --visualize run_output.json   # diversity_chart.png
python prepare.py --attractors run_output.json   # per-output attractor hits
```

## API notes

* Model: `claude-sonnet-4-6` ($3/$15 per MTok, 1M context, 64K max output)
* Prefilling assistant messages is NOT supported on Sonnet 4.6
* `temperature` and `top_p` are mutually exclusive
* Full baseline run (200 completions): ~$0.90
