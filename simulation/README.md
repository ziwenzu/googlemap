# Simulation and identification code

`anchor_transport.py` is a minimal executable check of the paper's exact-stability
algebra. It estimates category-relative selection factors from one anchor and
inverts a target observed histogram.

Example:

```bash
python3 simulation/anchor_transport.py \
  --anchor-observed .10,.10,.15,.25,.40 \
  --anchor-latent .20,.20,.20,.20,.20 \
  --target-observed .05,.10,.15,.30,.40
```

This is not the final estimator. Before empirical use, add:

1. multinomial audit and review sampling uncertainty;
2. zero-category handling declared before seeing results;
3. a joint optimizer for bounded selection-ratio drift and contamination;
4. simultaneous confidence sets across facilities;
5. clustered Monte Carlo coverage tests; and
6. mark-misclassification stress tests.

The four main DGPs are exact stability, bounded drift, cell-varying mark error, and
clustered contamination. Simulations validate algorithms and coverage; they do not
prove identification.
