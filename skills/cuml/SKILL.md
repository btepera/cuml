---
name: cuml
description: GPU-accelerate scikit-learn, UMAP, and HDBSCAN workloads with NVIDIA cuML. Use this skill whenever someone wants to run an sklearn/umap-learn/hdbscan workflow on a GPU, speed up a machine learning pipeline (training, clustering, dimensionality reduction, nearest neighbors, embeddings at scale), use cuml.accel or the cuml Python API, debug why cuML is slow or falling back to CPU, save/deploy cuML models, keep data on the GPU with cuDF/CuPy, or add optional cuML/GPU support to a library built on scikit-learn. Use it even if the user only says "make this ML code faster" and a GPU is available, or mentions RAPIDS.
---

# cuML

cuML is NVIDIA's GPU machine learning library (part of RAPIDS). It offers two ways in:

- **`cuml.accel`**: zero-code-change acceleration. It intercepts `sklearn`, `umap`, and `hdbscan` estimators and runs supported operations on the GPU, falling back to the original CPU implementation for anything unsupported. Start here whenever the user has existing sklearn/umap/hdbscan code.
- **The direct `cuml` API** (`cuml.ensemble.RandomForestClassifier`, `cuml.UMAP`, ...): sklearn-like estimators with GPU-specific options (e.g. multi-GPU kNN graph builds, batching, output types). Use it when the user needs control that the sklearn signature can't express, or keeps data on the GPU end to end.

## Pick the track

Read the reference file that matches the user's situation before answering:

| The user is... | Cues | Read |
|---|---|---|
| Accelerating their own script, notebook, or pipeline | "my script", "my notebook", "speed up training", "run this on GPU", embeddings/clustering at scale, deploying a trained model | `references/accelerating-workflows.md` |
| Adding GPU support to a library other people install | "my library", "our package", "our users", "optional dependency", "backend", "device= parameter", CI for a library | `references/library-integration.md` |

Supporting references (read when needed):
- `references/accel-compatibility.md`: which estimators `cuml.accel` accelerates and the exact conditions that cause CPU fallback. Check it whenever you review user code for GPU readiness.
- `references/api-differences.md`: constructor parameters that exist in sklearn/umap-learn/hdbscan but **not** in the direct cuML estimators (and vice versa). Check it before writing or fixing any direct `cuml.*` constructor call.
- `scripts/compare_params.py`: prints the same comparison live from the installed versions, if cuML and sklearn are importable. It's a tool for *you*; the user doesn't have it. If you want the user to check parameters themselves, point them to `help(cuml.X)` or `inspect.signature(cuml.X)` instead.

## Core facts (both tracks)

**Activating `cuml.accel`.** It must be enabled *before* sklearn/umap/hdbscan are imported:
- Script: `python -m cuml.accel script.py`
- Jupyter/IPython: `%load_ext cuml.accel` as the first cell
- Programmatic: `import cuml; cuml.accel.install()` before other imports
- Code you can't modify (CLI tools, third-party apps): `CUML_ACCEL_ENABLED=1 some-command`
- With cuDF's pandas accelerator: `python -m cudf.pandas -m cuml.accel script.py`, or `%load_ext cudf.pandas` then `%load_ext cuml.accel`

**Fallback is silent.** Unsupported estimators, hyperparameters, or inputs run on the CPU without error. So "it runs" doesn't mean "it ran on the GPU". (A few unsupported combinations raise an error instead; `references/accel-compatibility.md` lists them.) Review the code against that file, not against memory: the upstream docs and common knowledge are out of date in places (e.g. RandomForest `class_weight`/`sample_weight` and SVC `kernel="precomputed"` now run on GPU). Tell the user to verify:
- `python -m cuml.accel -v script.py` or `CUML_ACCEL_LOG_LEVEL=info`: logs each GPU call and fallback
- `python -m cuml.accel --profile script.py` / `%%cuml.accel.profile` / `with cuml.accel.profile():`: table of GPU vs CPU calls, time, and *why* each fallback happened
- `python -m cuml.accel --line-profile script.py` / `%%cuml.accel.line_profile`: per-line time and % on GPU (adds overhead; don't compare its timings)
- The profilers don't see work in subprocesses, so set `n_jobs=1` on `GridSearchCV`/`RandomizedSearchCV` etc. while profiling.

**The direct API is not signature-identical to sklearn.** Common traps: `n_jobs` exists only on `NearestNeighbors`/`KNeighbors*` (accepted and ignored); `RandomForest*` uses `split_criterion` (not `criterion`; regressor uses `'mse'`, not `'squared_error'`) and lacks `ccp_alpha`, `min_weight_fraction_leaf`, `warm_start`, `max_leaf_nodes` (it *does* support `class_weight`); `LogisticRegression` only supports `solver='qn'`; `DBSCAN`'s `algorithm` is `'brute'` or `'rbc'`; `HDBSCAN` only supports `metric='euclidean'`. Never guess a cuML parameter. Check `references/api-differences.md`. If the user wants to keep sklearn signatures, use `cuml.accel` instead.

**Data and dtypes.**
- Inputs: NumPy, pandas, cuDF, CuPy, anything with `__array__` or `__cuda_array_interface__`. Outputs mirror the input type by default (cuDF in → cuDF out). Override per estimator with `output_type=` (`'input'`, `'numpy'`, `'cupy'`, `'cudf'`, `'pandas'`), in a block with `with cuml.using_output_type('numpy'):`, or globally with `cuml.set_global_output_type(...)`.
- Prefer float32: half the memory, and many GPU kernels are fastest in it. Cast once up front.
- Features must be numeric; encode strings/categoricals first.

**Results are equivalent in quality, not bit-identical.** Different solvers and parallel floating-point order mean `coef_`, `labels_`, embeddings, etc. differ numerically from CPU. Compare with quality metrics (accuracy, R², ARI/AMI for clustering, `sklearn.manifold.trustworthiness` for UMAP/TSNE), never by exact equality. Some fitted attributes that are purely diagnostic aren't computed under `cuml.accel` (e.g. `LinearRegression.rank_`/`singular_`, `Lasso`/`ElasticNet.dual_gap_`, `affinity_matrix_`); check `references/accel-compatibility.md` before claiming a specific one is missing.

**When the GPU won't help.** Small data (thousands of rows, a few dozen features) usually isn't faster on GPU, and is often slower, because fixed costs dominate: CUDA initialization, host↔device transfer, and JIT compilation of some kernels on first call. Say so honestly and suggest CPU-side fixes first (`n_jobs=-1`, smaller search grids, `HalvingGridSearchCV`). Speedups grow with rows × features × algorithm cost; nearest neighbors, UMAP, TSNE, HDBSCAN, SVMs, and random forests on large data benefit most. When benchmarking, warm up first and exclude import time.

**Memory.** `cuml.accel` enables CUDA managed memory (UVM) when available, so it can spill past GPU memory into host RAM. That avoids OOMs but can be slow if heavily oversubscribed (`--disable-uvm` to compare). For the direct API, size the data first (rows × cols × 4 bytes for float32) and plan around GPU memory; see the embeddings section of `references/accelerating-workflows.md` for batching options.

**Scope.** Recommend single-GPU workflows, plus the direct-API multi-GPU kNN-graph option described in the workflows reference. Don't steer users toward distributed frameworks for scaling; it isn't what this skill covers.

## How to answer

1. Figure out the track and read its reference.
2. If the user shared code, audit it: which estimators/parameters/inputs will run on GPU, which fall back, and any plain bugs you notice along the way.
3. Give the smallest change that gets them onto the GPU (usually `cuml.accel`), then the targeted fixes for fallbacks, then how to verify (logging or profiler).
4. Use only parameters and functions you've confirmed exist, and check every kwarg in code you write for a `cuml.*` constructor against `references/api-differences.md`. In particular, never forward a dict of sklearn/umap-learn/hdbscan kwargs to a cuML class unfiltered: unknown kwargs raise `TypeError`, and unknown keys inside `build_kwds` are silently ignored. Correct code that runs on the CPU beats GPU code that crashes or quietly does the wrong thing.
