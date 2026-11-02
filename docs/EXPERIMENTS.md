# Experiments and statistical units

## Multi-source benchmark

| Dataset | Tasks | Classes | Window / stride | Recorded conditions |
| --- | ---: | ---: | --- | --- |
| PU | Four 3→1 | 3 | 5120 / 4096 | A:1500 rpm,0.1 Nm,1000 N; B:1500,0.7,400; C:1500,0.7,1000; D:900,0.7,1000 |
| PHM09 | Twenty 2→1 | 8 configurations | 5333 / 2666 | Ten speed/load domains, exact task list in `tasks.json` |
| JNU | Three 2→1 | 4 | 4000 / 2000 | 600,800,1000 rpm |

Run IDs are 0–9. PU retains the historical 18 bearings and uses the first 60 complete windows. Source train/validation roles are assigned by recordings, 16/4 per bearing/condition. PHM09 retains the recorded file split; validation class support can be incomplete. JNU uses separated time blocks from the recorded files. These are cross-condition protocols, not a claim of independent new-machine confirmation.

`protocols/source_roles.json.gz` contains the actual 270 MDG source task/run cases, their source roles and pairings. It does not contain waveform arrays. `protocols/tasks.json` names all 27 tasks. The detailed paper protocol is authoritative for domain and class meanings.

## Compared recipes

- Matched mechanism comparisons: IN+Mixup, unedited bank, endpoint timing, Q2-rate timing, impulse-only, global-moment matching and Q2. Half exposure and expected-energy shrinkage are sensitivity controls.
- Shared-backbone baselines: ERM, source-only DANN, source-only Deep CORAL.
- Author-implementation/adapted baselines: author CE, author IN+Mixup, CCDG, MOCD. They retain declared schedule/input adaptations, not identical original-paper protocols.
- Older identity/IN/event variants remain in the complete 22-recipe tables, but do not replace matched controls.

MOCD's equation-literal and prose interpretations remain distinct. The poor literal result is not used to characterize the original method's published performance. DA using unlabeled target data is not silently treated as DG.

## Separate extensions

PCR compares probabilities aggregated to the same PU recording roles, because native angular windows and short vibration windows differ. Its native frontend uses recorded target speed. Seeds 0–2 of the PCR archive have recording metrics only; do not present the window metrics available for seeds 3–9 as a complete ten-run PCR window comparison.

JNU SDG trains at each one source speed and evaluates the same frozen model at the two other speeds. The six directions arise from three source models per recipe/run. They are not six independently trained models. Q2's single-source analysis uses source-local contexts.

The full accounting is 5940 MDG + 240 PCR + 90 SDG model units = 6270. Different checkpoint policies and target roles yield 12720 metric records, not 12720 training repetitions.

## Aggregation

Within each dataset/run, tasks are averaged equally. Report the mean and sample SD (denominator 9) of the ten run scores. Dataset-equal pooled contrasts first compute each dataset/run score and then average the three datasets equally. Window accuracy, window macro-F1 and recording balanced accuracy are different metrics and are never interchangeable.

Both final and source-selected results are retained. The source-selection criterion is source-domain-equal window accuracy in the MDG experiments; the JNU SDG extension uses its declared balanced criterion. Run dispersion measures repeatability under these run protocols; it is not a direct noise robustness or machine reliability measurement.
