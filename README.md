# JAIDE

JAIDE is a foundation large language model built on the 5th root architecture paradigm, utilizing a Reversible Scatter Flow (RSF) stack of bijective, invertible cross-affine coupling layers — fundamentally distinct from all σ(W·x+b)-based architectures — which enables O(dim) memory complexity during backpropagation, hierarchical latent-space representation instead of next-token prediction, context preservation via versioned temporal graphs and surprise-based memory rather than KV cache, convergence-driven reasoning, and multi-layered gradient stability through spectral norm constraints, exp-range clipping, dynamic loss scaling, and a GradientFlowController.

Ranker-Processor Hybrid Long Context: Succinct Semantic Index (SSI) Processor: Reversible Scatter-Flow (RSF) Tokenization: Morpho-Graph Tokenizer (MGT) Optimizer: Spectral Fisher Diagonalizer (SFD)

Representation: Non-Sequential Information Representation (NSIR) Graph Model: Self-Similar Relational Graph (SSRG) Hardware: Relational Graph Processing Unit (R-GPU) Runtime: Chaos Core Kernel (CCK) Pipeline: Continuous Relational Extraction and Validation (CREV) Relational Optimizer: Entangled Stochastic Symmetry Optimizer (ESSO) Deployment: Fractal Network Deployment System (FNDS)

RSF implements triple-signal fused learning by computing prediction, reconstruction, and Jacobian log-det losses in one reversible Futhark GPU kernel that algebraically inverts layers for O(dim) memory, then fuses them into a single mixed-precision FP16/FP32 gradient updated via SFD Fisher conditioning, spectral normalization, and global clipping to enforce accuracy, invertibility, and symplectic geometry without extra networks or passes.

## Offline preparation

Two standalone tools prepare a corpus for the distributed trainer. Both are
plain Zig binaries with no Python, no PyTorch-style framework and no external
model code: the tokenizer, the trainer and the corpus format are all part of
this repository.

```sh
zig build vocab-train pretokenize -Doptimize=ReleaseSafe
```

### 1. Vocabulary (`jaide-vocab-train`)

```sh
zig-out/bin/jaide-vocab-train <corpus.jsonl> <vocab.bin> <vocab_size> [corpus_bytes]
```

* The corpus is JSONL; every record must carry a string `text` field.
* `corpus_bytes` bounds how much of the corpus is read (default 10 MiB). The
  byte-pair trainer is incremental, so a 10 MiB slice trains a 32 000 token
  vocabulary in a few seconds on one CPU core.
* Byte-pair merges are materialised from the bytes the tokens stand for, so no
  merge can ever produce a token whose text is a byte-token label. That keeps
  vocabulary encoding and decoding lossless across save/load cycles.
* All 256 single-byte tokens are always created first, so arbitrary UTF-8 input
  encodes without falling back to `[UNK]`.
* The vocabulary is built in a fixed order: four special tokens, the affix
  inventory, every byte token, the morphological roots mined from the corpus
  (`mineRoots`, at most `1 / JAIDE_VOCAB_ROOT_BUDGET_DIVISOR` of the free
  capacity, one eighth by default) and finally the byte-pair merges up to the
  requested size. The root set is a pure function of the corpus: stems are
  ranked by frequency and ties are broken by the stem text, so two runs produce
  byte-identical vocabularies.
* A stem becomes a root only when the corpus shows it in more than one shape:
  `JAIDE_VOCAB_ROOT_EVIDENCE=paradigm` (default) requires the stem to also occur
  as a whole word or behind at least two distinct affix combinations, so a
  single truncated word such as `surprise` can never mint the root `surpr`.
  `JAIDE_VOCAB_ROOT_EVIDENCE=frequency` mines every stem that reaches
  `JAIDE_VOCAB_MIN_ROOT_FREQUENCY` (default 2); it exists for corpora that repeat
  a fixed phrase list, where no paradigm can ever be proven, and it can produce
  truncations. When no stem passes the rule the tool says so instead of staying
  silent, and the morphological layer simply stays inactive.
* Every mined root is stored in the vocabulary's root table and marked as an
  anchor. Anchors are saved inside the vocabulary file, so every rank that loads
  the vocabulary has them, and the set is additionally exported next to it as
  `<vocab>.anchors`, a newline-separated token list in the format the
  distributed trainer reads through `JAIDE_TOKENIZER_ANCHORS`. No file is written
  when the vocabulary has no anchors, because the token-list reader rejects an
  empty file.
* The tool fails with `VocabularyTargetNotReached` when the corpus does not
  contain enough mergeable pairs for the requested size. Reaching an
  unreachable target silently would produce a vocabulary whose size no longer
  matches the token dataset header, so the mismatch is reported where the cause
  is still visible.
* Output is written to `<vocab>.tmp` and renamed, so a failed run never leaves a
  partially written vocabulary behind.

### 2. Token dataset (`jaide-pretokenize`)

```sh
zig-out/bin/jaide-pretokenize <corpus.jsonl> <dataset.bin> <vocab.bin> <context_length> [sample_count]
```

The output is the memory-mapped format the trainer reads (`JAIDETOK`, version 1,
64-byte header, little-endian `u32` payload):

| field | meaning |
| --- | --- |
| `total_tokens` | number of stored token ids |
| `vocabulary_size` | size of the vocabulary used to produce them |
| `maximum_sequence_length` | context length; every sample occupies `context_length + 1` tokens |

Operational properties:

* encoding is bounded to one window, so a pathological record cannot stall the
  process, and the longest-match scan is bounded by the longest vocabulary
  token instead of the remainder of the record;
* progress is reported on stderr every few seconds (`JAIDE_PRETOKENIZE_PROGRESS_SECONDS`),
  which makes a slow shard visible instead of looking like a hang;
* the output is written to `<dataset>.tmp` and renamed, so a failed run never
  destroys an existing valid dataset;
* `sample_count` is exact: if the corpus contains fewer non-empty records the
  tool fails instead of writing a short dataset.

## Benchmark pipeline

`scripts/modal_status_bench.py` orchestrates the whole run on Modal:

```sh
python scripts/modal_status_bench.py
```

The CPU phase (`prepare_cpu`) is responsible for everything that does not need a
GPU: it builds the vocabulary when the checkpoint volume does not have one,
imports or reformats an existing vocabulary, reuses a token dataset only when it
matches the current vocabulary and context length, and otherwise re-tokenizes
the corpus with parallel workers.

Each of these steps is bounded and observable:

* `JAIDE_BENCH_PRETOKENIZE_TIMEOUT` (default 7200 s) is the global limit of the
  parallel pre-tokenization;
* `JAIDE_BENCH_PRETOKENIZE_STALL` (default 300 s) is the stall watchdog: if no
  worker writes any output for that long, all workers are terminated and the
  run fails with the worker logs attached instead of idling on an expensive
  allocation;
* worker progress lines are streamed into the Modal log as they appear, so the
  run never goes silent;
* `JAIDE_BENCH_VOCAB_TIMEOUT` (default 1800 s) bounds vocabulary training and
  `JAIDE_BENCH_VOCAB_CORPUS_BYTES` (default 10 MiB) bounds how much of the
  corpus it reads.
* `JAIDE_BENCH_VOCAB_ROOT_EVIDENCE` (`paradigm` by default) selects the root
  mining rule the vocabulary trainer applies; the chosen value is logged with
  the vocabulary phase and recorded in the phase report.

The GPU phase then trains with the Futhark-accelerated RSF trainer and smokes the
inference endpoint. Training itself keeps its own watchdogs
(`JAIDE_BENCH_INIT_DEADLINE`, `JAIDE_BENCH_IDLE_WATCHDOG`) so a stalled step is
reported by name instead of waiting for the container timeout.

## Tests

```sh
zig build check              # semantic analysis of every module
zig build test-tokenizer     # tokenizer invariants: training, round-trip, persistence, bounded encoding,
                             # root mining, deterministic anchor list, anchor positions
zig build test-rsf           # RSF tensor, causal and spectral substrate
zig build test-rsf-native    # accel-free cross-module RSF invariants
```
