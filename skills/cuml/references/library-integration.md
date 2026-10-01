# Adding cuML GPU acceleration to a library

For maintainers of libraries built on scikit-learn / umap-learn / hdbscan who want users with NVIDIA GPUs to benefit, without making cuML a hard dependency or changing behavior for CPU-only users.

## The integration ladder

Present these as options with tradeoffs. They combine well, and most libraries should do (1) regardless.

### 1. Be `cuml.accel`-compatible and document it (zero library code)

Users run `python -m cuml.accel app.py`, `%load_ext cuml.accel`, or `CUML_ACCEL_ENABLED=1 your-cli ...`, and every sklearn/umap/hdbscan estimator your library constructs is accelerated where supported. Your job:
- Add a GPU CI job that runs your test suite under the accelerator (see Testing below) and fix what breaks.
- Document the activation methods in your docs/README.
- Know the limits: only estimator classes are intercepted. Standalone functions your library calls (e.g. `umap.umap_.nearest_neighbors`, `sklearn.metrics.pairwise_distances`, `sklearn.neighbors.kneighbors_graph`) run on CPU. If a hot path is a function, consider calling the estimator form instead.

### 2. Accept estimator objects (dependency injection)

Let users pass the component estimators: `MyModel(umap_model=..., cluster_model=...)`. GPU users pass `cuml.UMAP(...)` / `cuml.cluster.HDBSCAN(...)`; your library never imports cuML. Requirements on your side:
- Only use the public sklearn-style API on injected objects (`fit`, `transform`, `fit_predict`, `labels_`, ...).
- Convert outputs to NumPy at the boundary (`cupy.asnumpy`, `.to_numpy()`), or tell users to construct estimators with `output_type="numpy"`.
- Be aware of API differences (see `api-differences.md`); e.g. `cuml.cluster.HDBSCAN` may not provide extras like `exemplars_` or `outlier_scores_` that `hdbscan.HDBSCAN` users expect, so check with `hasattr` or document the limitation.

### 3. Explicit opt-in backend (`device=` / `backend=`)

For libraries that want a first-class switch. Sketch:

Prefer cuML's own sklearn → cuML conversion over a hand-written translation table. `cuml.X.from_sklearn(est)` works on an *unfitted* estimator: it maps parameters with the same rules `cuml.accel` uses (e.g. PCA `svd_solver` and `iterated_power='auto'`, random forest `criterion` → `split_criterion`, dropping CPU-only kwargs like `n_jobs`) and raises `UnsupportedOnGPU` when a setting has no GPU equivalent, which is your cue to stay on CPU. It returns an estimator with `output_type="numpy"`.

```python
import warnings

def _resolve_device(device):
    if device == "cpu":
        return "cpu"
    try:
        import cuml  # noqa: F401  (lazy: only imported on this path)
        import cupy
        has_gpu = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        has_gpu = False
    if device == "gpu" and not has_gpu:
        raise RuntimeError("device='gpu' requested but cuML / a CUDA GPU is not available. "
                           "Install with `pip install mylib[gpu]`.")
    return "gpu" if has_gpu else "cpu"          # device == "auto"

# CPU class -> cuML class with from_sklearn() support
_GPU_EQUIVALENT = {
    "PCA": "cuml.decomposition.PCA",
    "KMeans": "cuml.cluster.KMeans",
    "NearestNeighbors": "cuml.neighbors.NearestNeighbors",
    "IsolationForest": "cuml.ensemble.IsolationForest",
    "UMAP": "cuml.manifold.UMAP",          # from umap.UMAP
    "HDBSCAN": "cuml.cluster.HDBSCAN",     # from hdbscan.HDBSCAN
}

def _to_device(cpu_est, device):
    """cpu_est: an unfitted sklearn/umap/hdbscan estimator built from the user's params."""
    if device != "gpu" or type(cpu_est).__name__ not in _GPU_EQUIVALENT:
        return cpu_est
    import importlib
    from cuml.internals.interop import UnsupportedOnGPU   # internal path; may move
    mod, name = _GPU_EQUIVALENT[type(cpu_est).__name__].rsplit(".", 1)
    gpu_cls = getattr(importlib.import_module(mod), name)
    try:
        return gpu_cls.from_sklearn(cpu_est)
    except UnsupportedOnGPU as e:
        warnings.warn(f"{name}: falling back to CPU ({e})")
        return cpu_est

# usage
from sklearn.decomposition import PCA
pca = _to_device(PCA(n_components=32, svd_solver="randomized"), _resolve_device(device))
```

If you need a cuML-only option that has no CPU counterpart (e.g. UMAP `build_kwds`, `device_ids`), set it on the converted estimator (`gpu_est.set_params(...)`) after `from_sklearn`. Only fall back to a hand-written kwargs mapping for estimators without `from_sklearn`; in that case filter kwargs against `inspect.signature(gpu_cls)` *and* translate values (sklearn defaults like `iterated_power='auto'` are not valid cuML values).

Design points to raise:
- **Lazy imports only.** Never `import cuml` at module top level; CPU-only installs must not need it.
- **Packaging.** Offer an extra (`pip install mylib[gpu]`) that pulls the right wheel (`cuml-cu12` or `cuml-cu13`, matching the user's CUDA), or document conda installation. Don't add cuML to core dependencies.
- **Parameter translation.** cuML constructors are not sklearn-identical (see `api-differences.md`): `n_jobs` only on neighbors estimators; renamed (`criterion` → `split_criterion` for random forests, regressor `'squared_error'` → `'mse'`); missing (`warm_start`, `ccp_alpha`, umap-learn's `densmap`/`low_memory`/`n_jobs`, hdbscan's `memory`/`core_dist_n_jobs`, ...); restricted values (PCA `svd_solver` in `'auto'`/`'full'`/`'jacobi'`, `LogisticRegression` `solver='qn'` only, `HDBSCAN` `metric='euclidean'` only, `DBSCAN` `algorithm` in `'brute'`/`'rbc'`). `from_sklearn()` (above) handles this for supported estimators; when a user-specified value has no GPU equivalent it raises `UnsupportedOnGPU` and you fall back to CPU with a warning. Never pass a CPU kwargs dict to a cuML constructor unfiltered, and don't silently drop parameters that change results.
- **Output types.** cuML mirrors the input type (CuPy in → CuPy out). Set `output_type="numpy"` on each estimator, or wrap calls in `with cuml.using_output_type("numpy"):`, so your library's return types are stable. Avoid `cuml.set_global_output_type` in a library; it mutates global state for the user's whole process.
- **Inputs.** cuML accepts NumPy, pandas, CuPy, cuDF. Cast to float32 for speed/memory where your library's semantics allow. On the CPU path, convert CuPy/cuDF inputs to host first (`cupy.asnumpy`, `.to_pandas()`): sklearn won't accept them.
- **Results differ numerically** from the CPU path. Document it; test with tolerances/quality metrics.
- **Consider whether (1) is enough.** A hand-written dispatch layer is code you maintain against two moving APIs. `cuml.accel` handles parameter checks and CPU fallback for you; (3) is worth it when you need GPU-only features (batching, `device_ids`, approximate kNN) or want an explicit user-facing switch.

## Don't call `cuml.accel.install()` on users' behalf

Calling it at library import (or implicitly when a GPU is detected) has process-wide side effects the user didn't ask for:
- It patches `sklearn`, `umap`, and `hdbscan` for **all** code in the process, not just your library.
- It sets `CUML_ACCEL_ENABLED=1` in the environment, so **subprocesses** (joblib workers, CLI calls) get accelerated too.
- It may switch the global RMM memory resource to **managed memory**, affecting other GPU libraries in the process.
- It only affects modules imported **after** it runs, so behavior depends on import order.

Let users opt in themselves (CLI, `%load_ext`, env var, or their own `cuml.accel.install()` call). If you want a convenience, expose an explicit function your users call, and document its global effect.

## Testing under cuml.accel

Run the existing suite with the accelerator loaded as a pytest plugin (installs before test collection):

```bash
pytest -p cuml.accel tests/
CUML_ACCEL_LOG_LEVEL=info pytest -p cuml.accel tests/ -k some_test   # see GPU vs fallback per call
```

Track known failures with `--xfail-list` (a YAML file that the plugin applies as xfail markers):

```yaml
- reason: "cuml.accel does not compute n_iter_"
  strict: false              # optional, default true
  condition: "scikit-learn>=1.8"   # optional version condition
  tests:
    - "tests.test_models::test_n_iter_reported"
```

```bash
pytest -p cuml.accel --xfail-list=cuml_accel_xfails.yaml tests/
```

Test IDs are `module::test_name`. The plugin warns about entries that match no test.

CI layout: keep the CPU-only job unchanged (no cuML import anywhere), and add a separate job on an NVIDIA GPU runner with cuML installed.

## What typically breaks under cuml.accel

- **Exact-equality assertions on fitted values** (`coef_`, `labels_`, `components_`, embeddings, tree structure). GPU solvers differ. Assert on quality metrics or use tolerances; for clustering, compare ARI/AMI.
- **Fitted attributes not computed** by the GPU path (e.g. `LinearRegression.rank_`/`singular_`, `Lasso`/`ElasticNet.dual_gap_`, `affinity_matrix_`, `IsolationForest.estimators_samples_`; `hdbscan.HDBSCAN` `exemplars_`, `outlier_scores_`, `relative_validity_`; see `accel-compatibility.md`). Accessing them raises `AttributeError`.
- **Private attributes.** Accelerated estimators raise `AttributeError` for `_`-prefixed attributes (`est._tree`, solver internals). Use public API.
- **Error and warning messages** differ (and may include C++ traces); `pytest.raises(match=...)` / `pytest.warns(match=...)` on exact text can fail.
- **Determinism:** TSNE isn't fully deterministic even with `random_state`, and HDBSCAN can differ run to run when the data has duplicate points. UMAP with `random_state` set is reproducible (with the default `build_algo='auto'`, the seed forces the brute-force kNN build); only an explicit `build_algo='nn_descent'` plus a seed is non-deterministic.
- **Type introspection:** `isinstance(est, sklearn.X)` still works (the proxy class forwards instance/subclass checks), and `type(est).__qualname__` matches sklearn's. But `type(est).__module__` is the *public* import path (e.g. `sklearn.ensemble`), not sklearn's private defining module (`sklearn.ensemble._forest`), so exact checks against private module paths break (`startswith("sklearn")` still works). Identity comparisons against a class captured before acceleration can also differ. Use `cuml.accel.is_proxy(est)` to detect an accelerated estimator explicitly, and `cuml.accel.enabled()` to check whether the accelerator is active.
- **Performance traps:** reading public fitted attributes copies them back from the GPU; doing this in tight loops can erase the speedup.
- **Serialization:** models pickled under `cuml.accel` load as plain sklearn objects without it, which is fine. But persisted objects and cache formats should be tested both ways.
