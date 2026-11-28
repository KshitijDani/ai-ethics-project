# Multi-turn response analysis

## Intent

Compare departure from baseline for each persona, question/turn, topic,
personalized mode, and generating model. Produce both methods so the researcher
can compare their scores and plots. Approved for implementation in chat on 2026-11-28.

## Approaches and selected scope

The alternatives discussed were a mean baseline embedding and all response
pairs. The user selected both, exposed through two separate scripts:

- `analysis/multi_turn_response_analysis_average.py`
- `analysis/multi_turn_response_analysis_pairing.py`

A shared internal module handles CSV validation, embedding, grouping, and
plotting so the scripts differ only in their scoring method. The existing
placeholder can become that shared module. A single method flag was considered
but separate commands match the requested workflow. Duplicating the entire
pipeline would make methodological comparison vulnerable to unrelated changes.

## Inputs and matching

Each invocation accepts a results root (the current default is `results/`) or
an explicit list of input CSVs. Recursively discover response CSVs beneath the
root and read model/mode/persona from CSV columns rather than paths or filenames.
The uploaded conditions occupy separate experiment folders and the once-only
files have no `multi-turn` parent. Keep generated analysis outputs outside the
input root; validate the response schema and reject ambiguous duplicate inputs.
Automatically discover models, topics, personas, and personalized modes.
Compare only within the same model, topic, question ID, turn, and question text.
Reject duplicate response identities, blank responses, mismatched questions,
and missing baseline matches. Report run counts and incomplete coverage rather
than assuming every experiment has five complete runs or ten turns. Retain
identical response texts from distinct runs: these are valid repeated outcomes,
not duplicate records. Cache their embeddings without dropping observations.

Embed only the assistant response, using one embedding model and preprocessing
policy across all conditions and both scripts. Unit-normalize each response
vector. Reuse a cache keyed by response text and embedding configuration.

## Method A: average baseline

For one model/question, let `b_j` be the unit baseline vectors and `p_i` the
unit vectors for one persona/mode. Compute `m = mean(b_j)`, then the normalized
baseline centroid `c = m / norm(m)`. Score every persona response as
`dot(p_i, c)` and summarize its mean and descriptive standard deviation.
Reject a zero or near-zero centroid with a clear diagnostic.

## Method B: response pairing

Score every persona-to-baseline pair as `dot(p_i, b_j)`. Export the individual
pairs and summarize their mean and descriptive standard deviation. Five persona
runs and five baseline runs produce 25 pairs; run IDs are retained for audit.
These are all pairs, not a same-run-number pairing.

## Shared outputs

Write results to separate method-specific directories containing per-response
or per-pair scores, per-question summaries, PNG plots, and a manifest recording
inputs and embedding configuration. Source CSVs may be enriched with embedding
columns under the user's subsequent request; preserve all original fields.

One main graph per generating model, topic, and personalized mode shows turn
on the x-axis and cosine similarity on the y-axis, with one line per persona.
Use the same axis limits and persona colors for both methods. Include a
horizontal perfect-similarity reference at 1 and an actual baseline consistency
line: mean cosine over distinct baseline-to-baseline pairs, excluding self-pairs.
Export within-persona consistency over distinct run pairs as a separate metric.

Bands, if displayed, represent descriptive score variation and are labeled
explicitly; they are not confidence intervals. Omit undefined spread or
consistency statistics when there are insufficient observations.

## Interpretation and comparison

### Research hypothesis: weakening personalization across turns

The researcher hypothesizes that, despite conversation history retaining the
persona context, personalization may weaken over successive turns and responses
may move toward the model's default behavior. The proposed interpretation is a
return toward the model's own learned social values rather than continued
adherence to the supplied persona. This is a hypothesis to investigate, not an
established mechanism or conclusion.

Distinguish two directions explicitly: departure from the personalized context
would, under this hypothesis, appear as increasing similarity to the matched
unpersonalized baseline. Increasing departure from baseline would indicate the
opposite direction on this metric. A response can also change without moving
toward baseline, so neither direction fully measures persona adherence.

Retained history makes an influence of earlier context on later responses
possible. It does not identify the cause of an observed trend: question content
and order change with turn, and cosine similarity does not directly measure
history use, persona adherence, or social values. Treat baseline as an empirical
reference for unpersonalized behavior under the experiment's system prompt and
generation conditions, not a direct measurement of the model's social values.

Compare the once-only and repeated-persona conditions as evidence relevant to
the hypothesis. If the once-only condition moves toward baseline more than the
repeated-persona condition, that would be compatible with weakening persona
influence without reinforcement. It would not by itself demonstrate that the
model ignored history or establish a causal effect of turn count. Stronger
tests would require additional controls, such as varying question order or
placing the same target question after different controlled histories, and
independent measures of persona adherence. Those experiments are outside the
current two-script implementation.

### Reporting boundaries

- Departure from baseline and consistency across repetitions are separate.
- Different turns ask different questions, so a trend cannot isolate a causal
  effect of conversation length from question content/order.
- Embedding similarity does not by itself establish recommendation identity,
  persona fidelity, or the absence/presence of bias.
- With unit response vectors, mean all-pairs similarity equals mean centroid
  similarity multiplied by `norm(m)`. Thus the difference in these mean scores
  directly reflects baseline concentration; the methods are related summaries,
  not independent evidence. Export `norm(m)` to make this visible.
- Do not use the number of pairs as an independent sample count for inference.

## Approved implementation choices

Use the existing Python/pandas environment, NumPy for scoring, Matplotlib for
static plots, and a configurable local Sentence Transformers encoder. Approved
starting encoder: `sentence-transformers/all-mpnet-base-v2` for English responses.
Its model card documents truncation beyond 384 word pieces; detect and reject
overlength inputs by default, with any later chunking policy made explicit.
These implementation choices were included in the design approved in chat.

## Validation and non-goals

Use small known-vector fixtures to verify both metrics and their mathematical
relationship, plus synthetic CSVs for matching, counts, and output generation.
Inspect generated plots before claiming completion. No new simulations, API
uploads, hypothesis tests, confidence intervals, or causal claims are included.

## Inspection of uploaded data — 2026-11-28

Fetched `origin` and fast-forwarded the `scripts` branch from `8b68788` to
`2f35a5a` (`origin/main`). The remote default is `main`; no `master` branch exists.
Existing local edits were preserved. No analysis implementation was added.

Nine response CSVs are tracked under `results/`:

- Baseline: `full-baseline/multi-turn/baseline/<model>.csv`.
- Repeated persona: `full-prompt-prepend/multi-turn/prompt-prepend/prompt-prepend <model>.csv`.
- Once-only persona: `Prompt-Prepend-Once <model>.csv` directly under `results/`.

Inspection using pandas confirmed:

- 4,200 rows: 600 baseline, 1,800 prompt-prepend, 1,800 prompt-prepend-once.
- Models: gemma, qwen, ministral. Personalized conditions contain American,
  Chinese, and French personas. Topics: books, food, hobby, movies.
- Every condition/question has five distinct run numbers and every topic has
  ten turns. Question IDs and exact question text agree across all conditions
  and models. No blank responses or duplicate response identities were found.
- All nine files use the existing 13-column simulator schema.
- First-turn persona prefixes are consistent within each persona/mode. All
  1,620 later once-only user messages equal the question; none of the 1,620
  later repeated-persona user messages equal the question alone.
- Exact repeated answers occur across runs; some groups have only one distinct
  response among five runs. Preserve these and allow zero descriptive spread.
- Median response lengths in characters by baseline/repeated/once-only mode:
  Gemma 31.5/39/33; Qwen 160/179/179.5; Ministral 236.5/322.5/298.
  The longest response is 1,148 characters. Character counts do not establish
  tokenizer coverage; actual encoder token lengths still need checking.

These observations change input discovery and reinforce the need to interpret
full-response similarity carefully: explanatory wording and response length may
contribute to scores alongside the recommendation itself. Keep full responses
as the proposed initial measurement; do not silently extract recommendation
titles or remove explanations. Any recommendation-only comparison is a separate
future design choice.

Conditions were generated in separate timestamp ranges. The response CSVs omit
generation settings, model file versions, and system-prompt metadata. Record
source CSV hashes and available provenance in the analysis manifest, but do not
claim these settings were identical merely because the row schema matches.

## Approved amendment: embeddings in source CSVs

The user additionally requested persistent embeddings directly in each response
CSV, so subsequent analyses do not repeat embedding work. Add `embedding_vector`
(a JSON array) and `embedding_metadata` (encoder configuration/revision, token
count, and response/vector/configuration checksums). Reuse valid vectors without
loading an encoder when all responses are covered. Regenerate or recover stale,
incompatible, or damaged entries. Retain the separate revision-aware cache.

Preserve original column values, additional freeform columns, and row order;
write files atomically and check source hashes immediately before replacement.
Coordinate enrichment writers with per-source locks. A concurrent source change
must abort enrichment rather than be overwritten. Record guarded pre/post file
hashes in output manifests. Source enrichment is the explicit exception to the
original no-source-mutation constraint. Publish analysis output directories only
after all their tables, plots, and manifest are complete.

Source for encoder properties:
https://huggingface.co/sentence-transformers/all-mpnet-base-v2
