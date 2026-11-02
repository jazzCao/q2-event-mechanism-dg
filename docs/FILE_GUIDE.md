# File responsibilities

## Small scientific API

| File | Read it for |
| --- | --- |
| `q2/operator.py` | Observed-unit edit and crop-support checks; the shortest entry to Q2. |
| `q2/renderer.py` | Finite-support zero correction used in the experiment. |
| `q2/response.py` | Continuous event coordinates and fixed analytic response kernels. |
| `q2/inverse.py` | Harmonic/stationary dictionary, proximal updates and CPU sparse inverse reference. |
| `q2/frontend.py` | Exact 2048-bin log-FFT calculation. |
| `q2/network.py` | CNN/IN and classifier, using original state-dictionary names. |
| `q2/sampling.py` | Common edit mask, random sampling, initialization seed and six-B mean loss. |
| `q2/batch.py` | Compact source feature/bank tensor adapter for Q2 and its two direct controls. |
| `q2/controls.py` | Impulse-only and moment-matching controls. |
| `tests/test_operator.py` | Analytic properties and reproducible random draws, without dataset access. |

## Results and paper

`evidence/ten_run_results.json.gz` contains all model-role metrics. `analysis.json.gz` contains additional derived paired comparisons and extensions. `paper_table_audit.json` maps paper table values to recipes, datasets, policies and metrics. `audit_index.json` identifies the original final audit files. The independent reviewer command recomputes results instead of trusting the PASS field.

`paper/main.tex` is the manuscript entry. `paper/generated/method.tex`, `protocol.tex` and `implementation.tex` contain technical sections. Other generated `.tex` files are tables or source-example inputs. PDF figures are vector inputs; SVGs are editable; PNGs are previews. `framework_publication` is the overall architecture and `mechanism_publication` is the analytic operator illustration.

## Provenance

`provenance/BUILD.json`: packaging origin and scope. `EXTRACTED_DEFINITIONS.json`: exact function/class sources. `ORIGINAL_CODE_INDEX.json`: the frozen implementation archive and lock matches. `CODE_MAP.json`: main original implementation entry points. `FILES.json`: distributable file integrity. `original_code/INDEX.md`: one description per archived source file.

## Material intentionally excluded

No dataset signals, large models, probabilities, author control ledgers, abandoned candidate reports, IDE state, private conversation exports, third-party repository clones or literature PDFs are included. These are retained separately in the author's working repository when needed for provenance. Their absence from this reader-facing directory does not mean they were deleted from the research record.
