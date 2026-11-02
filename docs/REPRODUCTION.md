# What can be reproduced from this package

## 1. Published-number verification — standalone and checked

`python tools/reproduce_results.py` requires only Python's standard library. It independently groups all saved task/run records, reconstructs recording metrics from saved confusion matrices, checks all 450 original summary entries and emits a CSV plus paired contrasts. It does not reuse the saved summary values as its calculations.

`python tools/check_integrity.py` verifies every distributed input against `provenance/FILES.json`. This proves file identity, not the truth of the scientific hypothesis. Outputs and bytecode are excluded from the release manifest.

## 2. Core mathematical implementation — standalone software checks

Install `requirements-core.txt` and run `python -m unittest discover -s tests -v`. Tests cover the analytic renderer against an independent direct kernel sum, the exact identity transform, retained remainder, crop support, common edit mask and random-state replay. They use small invented arrays and are not new PU/PHM09/JNU results.

The CPU inverse reference and the classifier are available for inspection. Install `requirements-model.txt` to import the PyTorch classifier/batch definitions. `Net(classes, 'E0_EVENTS')` is the Q2 deployed architecture; `risk_loss(torch, logits, labels, permutation, lam, False)` is the matched six-B mean risk. **Do not call `Net.losses` as the Q2 training loss:** that legacy class method is for an older two-B layout and is retained only as part of the exact extracted class definition.

## 3. Full retraining — original implementation supplied, preparation required

The paper results were produced by the frozen original implementations, not by a new portable runner in this package. `original_code/INDEX.md` indexes 131 original source files and `provenance/ORIGINAL_CODE_INDEX.json` records their experiment-lock hashes. They deliberately retain original code bytes, including local paths and old run guards.

A new machine requires:

1. Obtain PU, PHM09 and JNU recordings through their providers. Do not redistribute the local literature library or dataset files through this package.
2. Recreate recording identities, class labels, windows and source roles from the protocol, matching the supplied role manifests. More windows must not change a recording's role.
3. Generate the same log-FFT features and source-only inverse/context banks. Source pairs must be legal for the current task and seed; do not reuse a context belonging to validation or target just because it exists in a cache.
4. Supply these arrays to the original training mathematics, retaining batch layout, initialization, optimizer, random streams, BN and source model selection. Use new output locations; do not rebind or modify historical locks.
5. Freeze the intended source-trained comparison before target evaluation; aggregate with the same metric definitions and ten-run weighting.

The historical campaign's queue/resume scripts are not general-purpose commands. Their dates, local SQLite ledgers and D-drive caches must not be replayed as a new experiment. A fresh portable campaign driver and raw-data adapters should be separately checked before advertising one-command full retraining. Neither extraction parity nor existing audit PASS proves such a full rerun on a new computer.

There is no scientific reason to launch another 6270-model campaign merely to package this repository. A future release-driver parity check should use bounded software/engineering checks, preserve the original results, and identify precisely what equivalence it establishes.

## Data availability

The paper's references identify the Paderborn University bearing benchmark, PHM Society 2009 challenge and Jiangnan University bearing data. Local dataset paths and complete cache pointers were removed from distributed result records; array SHA-256 bindings were retained. The author workspace preserves the original records for provenance. Trained weights and full per-window probabilities are not in this compact package; adding a selected artifact later requires a separate file manifest and data-availability decision.

## Manuscript compilation

The `paper/` directory includes its CAS template and generated figure/table inputs. Run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice from `paper/`; references are maintained inline. A compiled PDF is supplied for readers without LaTeX. The author machine used TeX Live 2026. Do not run historical manuscript generators that overwrite `paper/`.
