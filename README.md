# Observation-anchored event-mechanism editing (Q2)

Reviewer companion to the manuscript by **Runze Cao and Jie Liu** (corresponding author), School of Reliability and Systems Engineering, Beihang University (BUAA), Beijing, China.

Start with [the manuscript](paper/main.pdf), [the method guide](docs/METHOD.md), and [the experiment guide](docs/EXPERIMENTS.md). The paper is a draft; it has not been accepted or assigned a DOI. Q2 is the current method. The earlier R3/CNSR methods are historical experiments, not aliases for Q2.

## Verify the results in two commands

From this directory, with Python 3.10 or newer:

```sh
python tools/check_integrity.py
python tools/reproduce_results.py
```

These commands need only the Python standard library. They verify the distribution and recompute ten-run means, sample standard deviations and paired contrasts from all 12,720 saved metric roles. Results go to `outputs/statistics/`; no datasets, checkpoints, GPU, training or model predictions are required.

For analytic software checks of the operator:

```sh
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
```

The small synthetic checks are software checks, not additional experimental evidence. See [reproduction levels and requirements](docs/REPRODUCTION.md) before attempting a new training run.

## Directory guide

| Location | Purpose |
| --- | --- |
| `paper/` | Self-contained LaTeX draft, compiled PDF, tables and editable vector figures. |
| `q2/` | Readable scientific components: inverse model, event renderer, edit operator, log-FFT, classifier, sampling and loss. |
| `protocols/` | The 27 MDG task definitions and 270 source task/seed role manifests. No waveforms. |
| `evidence/` | All ten-run metric records, saved confusion matrices, derived statistics and paper-table references. Local cache paths are removed. |
| `tools/` | Portable file integrity and statistical reproduction commands. |
| `tests/` | Analytic operator, support, identity, remainder and sampler checks. |
| `original_code/` | Exact original source files identified by experiment locks; an indexed implementation archive, not a new training launcher. |
| `provenance/` | File hashes, extraction locations, original code bindings and packaging environment. |
| `docs/` | Method, data protocol, reproduction, file guide and release limitations. |

## What the numbers mean

The main panel has 4 PU, 20 PHM09 and 3 JNU multi-source tasks, ten runs per task, and 22 complete recipes. PCR recording-level comparisons and six JNU single-source directions are separate extensions. Fixed-final and source-validation-selected checkpoints are reported separately. Multiple target evaluations/checkpoints of the same trained model are not independent training repetitions.

The proposed recipe has window accuracies of 83.12±0.94%, 71.98±0.95% and 99.67±0.17% on PU, PHM09 and JNU. Its dataset-equal increment over matched IN+Mixup is 0.80 percentage points. All controls and unfavourable comparisons are retained; this distribution does not claim that Q2 is best on every dataset or metric.

## Data and code status

Raw datasets, trained weights, full probability arrays, local control databases, third-party repository clones and literature PDFs are not bundled. Obtain datasets from their respective providers and respect their terms. [THIRD_PARTY.md](THIRD_PARTY.md) distinguishes our components, related methods and external material.

This is a reviewer package with verified statistics and extracted scientific components. The original full training campaign used locally bound data/cache paths, numerical source banks and recovery ledgers. This package does **not** claim a freshly validated, one-command, raw-data-to-all-results training pipeline. The full implementation remains inspectable in `original_code/`; [REPRODUCTION.md](docs/REPRODUCTION.md) states exactly what is portable and what requires preparation.

A project-wide open-source licence and author contact details remain to be supplied before a public release. No automatic upload or public repository visibility change is performed by these commands.
