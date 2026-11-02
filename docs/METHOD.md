# Method and code map

Q2 performs source-only event editing before diagnostic training. It does not estimate a condition branch and subtract that branch at inference.

1. **Source analysis:** two eligible source-training contexts determine a finite response dictionary, sparse nonnegative event coefficients, and harmonic/stationary terms. `q2/inverse.py` exposes the original NumPy reference definitions. Historical production used the CUDA solver/graph implementation in `original_code/`.
2. **Event rendering:** `q2/response.py` evaluates the same analytic response kernels at transformed event coordinates; `q2/renderer.py` restores exact zeros outside finite support. Event locations move about the context centre. Response kernels are retained.
3. **Observation-anchored edit:** `q2/operator.py` calculates `x + scale * (q*u_q - u_1)`. The two Q2 rates are `exp(±log(1.5)/sqrt(3))`. At rate one the output is the original observation. The remainder `x - scale*u_1` is retained, rather than classified as disposable condition noise.
4. **Frontend:** `q2/frontend.py` applies log(1 + FFT magnitude) and linear interpolation to 2048 bins. It is not a learned frequency mask.
5. **Diagnostic learning:** `q2/network.py` contains the original Table-3 CNN and head definitions. Feature IN follows the convolution stack. `q2/batch.py` exposes factual and auxiliary views for IN+Mixup, unedited bank and Q2; `q2/sampling.py` holds the matched sampling and loss functions.
6. **Deployment:** log-FFT, CNN/IN and the head consume a vibration window. No source inverse analysis, paired recording, target-domain fitting or extra learned branch is needed at deployment.

The source pair is used to estimate a source observation mechanism. It is not a pair required at test time. Labels and source conditions are used in training and legal pairing. No new physical identity should be inferred from dataset grouping labels.

## Fixed matched training recipe

- 2000 Adam updates, learning rate 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay zero, AMP disabled.
- Domain/class-balanced sampling, nominal batch 120; actual B is 117/112/120 on PU/PHM09/JNU.
- Edit mask probability 0.5, shared across source-domain slots. Independent same-class bank samples are permitted.
- Mixup Beta(0.2,0.2) on the auxiliary log-FFT views.
- One forward concatenates three factual copies and three mixed copies; ordinary BN sees the same 6B layout in matched arms. The loss is one-half factual CE plus one-half mean mixed CE.
- Source checkpoints 0/1000/2000; primary reporting at 2000, source-selected reporting separately.

The endpoint timing control uses rates 2/3 and 3/2; Q2 uses the interior rates. The compact `q2.operator.query_features` output has only the two Q2 rates. The historical four-rate bank has Q2 in positions 2 and 3. Do not confuse these array conventions.

`provenance/EXTRACTED_DEFINITIONS.json` binds each extracted definition to its original file, line numbers, source hash and AST hash. New wrappers in `operator.py` and `batch.py` are packaging interfaces; their checks do not constitute a rerun of the full classifier experiment.

The event coefficients are numerical estimates, not direct physical measurements. The strength/rate coupling is the specified augmentation rule, not an inferred universal mechanical law. Consult the paper for the experimental contribution and limits of that model-based intervention.
