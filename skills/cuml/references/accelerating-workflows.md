# Accelerating your own ML workflows with cuML

For users running their own scripts, notebooks, or pipelines. Sections:

1. [Choosing cuml.accel vs the direct API](#1-choosing-cumlaccel-vs-the-direct-api)
2. [Getting an existing sklearn script onto the GPU](#2-getting-an-existing-sklearn-script-onto-the-gpu)
3. [Notebooks and pandas-heavy pipelines](#3-notebooks-and-pandas-heavy-pipelines)
4. [Keeping data on the GPU (cuDF / CuPy)](#4-keeping-data-on-the-gpu-cudf--cupy)
5. [Embeddings at scale: UMAP + HDBSCAN](#5-embeddings-at-scale-umap--hdbscan)
6. ["It's barely faster": diagnosing with the profiler](#6-its-barely-faster-diagnosing-with-the-profiler)
7. [When not to use a GPU](#7-when-not-to-use-a-gpu)
8. [Saving, loading, and deploying models](#8-saving-loading-and-deploying-models)

---

## 1. Choosing cuml.accel vs the direct API

| Situation | Use |
|---|---|
| Existing sklearn / umap-learn / hdbscan code, want it faster with minimal changes | `cuml.accel` |
| Notebook or code shared with people who may not have a GPU | `cuml.accel` (code stays plain sklearn) |
| Data already lives on the GPU in cuDF/CuPy and should stay there | direct `cuml` API |
| Need GPU-only options: batched kNN graph builds, multi-GPU `device_ids`, approximate neighbor algorithms, `output_type` control | direct `cuml` API |
| Need an sklearn hyperparameter cuML doesn't support | `cuml.accel` (falls back to CPU for that estimator only), or change the hyperparameter |

Both can coexist: under `cuml.accel` you can still import and use `cuml.*` estimators directly for the parts that need GPU-specific control.

## 2. Getting an existing sklearn script onto the GPU

1. Run it unchanged: `python -m cuml.accel train.py`.
2. Audit the code against `accel-compatibility.md`. Look for:
   - Estimators that aren't accelerated at all (e.g. `HistGradientBoosting*`, `SimpleImputer`, `GradientBoosting*`). They run on CPU as usual.
   - Hyperparameters that trigger fallback. Frequent ones:

     | Fallback trigger | GPU-friendly alternative |
     |---|---|
     | `RandomForest*(criterion='log_loss')` (clf) / `'absolute_error'`, `'friedman_mse'` (reg) | `'gini'`/`'entropy'` (clf), `'squared_error'`/`'poisson'` (reg) |
     | `RandomForest*` with `class_weight='balanced_subsample'`, `warm_start`, `ccp_alpha`, `min_weight_fraction_leaf`, integer `max_samples`, NaNs in `X` | `class_weight='balanced'` (runs on GPU); drop the others; float `max_samples`; impute NaNs first. Note `class_weight='balanced'`/dict and `sample_weight` **do** run on GPU, despite older docs |
     | `LogisticRegression(warm_start=True)` or non-default `intercept_scaling` | drop them |
     | `SVC(probability=True)` | Depends on what the user needs: if only labels or decision scores, drop `probability=True`; if probabilities matter, consider a GPU-accelerated classifier with `predict_proba` (`LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`), or calibrate the SVC (`CalibratedClassifierCV(SVC(), ensemble=False)`). Offer the options; don't prescribe one |
     | Callable `kernel` on `SVC`/`SVR` | a string kernel, or `kernel='precomputed'` with a precomputed Gram matrix (runs on GPU) |
     | `SVC` on multiclass `y` | still works (CPU); consider `LinearSVC` or a one-vs-rest wrapper if speed matters |
     | `Ridge(solver='lbfgs')` or `positive=True` on linear models | default solver; drop `positive` unless required |
     | `OneHotEncoder` with `handle_unknown='infrequent_if_exist'`/`'warn'`, `min_frequency`, `max_categories`, `drop='if_binary'` | `handle_unknown='ignore'`; do rare-category grouping in pandas/cuDF beforehand |
     | **Sparse output feeding a model that needs dense input.** `OneHotEncoder` defaults to `sparse_output=True`, and `ColumnTransformer` keeps the result sparse when overall density is below `sparse_threshold` (0.3). A downstream `RandomForest*`, `KMeans`, `DBSCAN`, linear SVM, etc. then gets sparse `X` and falls back to CPU, often the most expensive step in the pipeline | `OneHotEncoder(sparse_output=False)` (or `ColumnTransformer(sparse_threshold=0)`) when the dense matrix fits in memory |
     | `StandardScaler`/`MinMaxScaler`/`MaxAbsScaler` with scikit-learn < 1.8 | upgrade scikit-learn to ≥ 1.8 |
     | `TSNE(n_components=3)` | 2D only on GPU |
     | `PCA(n_components='mle')` | pick an integer (a float variance ratio doesn't fall back but isn't handled like sklearn) |
     | Unsupported `metric` on neighbors / DBSCAN / HDBSCAN | see the metric lists in `accel-compatibility.md` |
     | Sparse `X` on RF, KMeans, DBSCAN, SpectralClustering, TruncatedSVD, linear SVMs | densify if it fits, or accept CPU |

   - Plain bugs. Point them out even if they aren't GPU-related.
3. Tell the user how to confirm: `python -m cuml.accel -v train.py` (logs `... ran on GPU` / fallback reasons) or `--profile`.
4. Mention that fitted values differ numerically from CPU, so validate with metrics, not `coef_` equality.

## 3. Notebooks and pandas-heavy pipelines

- First cell, before any other imports:
  ```python
  %load_ext cudf.pandas   # optional: GPU pandas
  %load_ext cuml.accel
  ```
  Loading after `import sklearn` / `import pandas` does nothing for already-imported modules; restart the kernel.
- `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `RandomizedSearchCV`, and `cross_val_score` all work; the accelerated estimators inside them run on GPU.
- Many workers hitting one GPU (`n_jobs=-1` in a search) can contend for GPU memory and compute. If GPU fits are fast, `n_jobs=1` or a small number is often as fast or faster. Measure.
- Check what ran where with a profiled cell: `%%cuml.accel.profile` at the top of the cell.

## 4. Keeping data on the GPU (cuDF / CuPy)

When data is produced on the GPU (cuDF ETL, CuPy, PyTorch tensors), avoid round-tripping through host memory:

```python
import cudf
from cuml.model_selection import train_test_split
from cuml.decomposition import PCA
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score

gdf = build_features()                         # cudf.DataFrame, all-numeric
X = gdf.drop(columns=["label"]).astype("float32")
y = gdf["label"].astype("int32")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
pca = PCA(n_components=64).fit(X_tr)
X_tr, X_te = pca.transform(X_tr), pca.transform(X_te)   # transform both splits
clf = RandomForestClassifier(n_estimators=200, max_depth=16).fit(X_tr, y_tr)
preds = clf.predict(X_te)                       # cuDF Series, still on GPU
print(accuracy_score(y_te, preds))             # only a scalar comes back to host
```

Key points:
- **Swap sklearn helpers for cuML ones.** sklearn's `train_test_split` and `sklearn.metrics` won't accept cuDF objects (cuDF refuses implicit host conversion), which is usually why `.to_pandas()` was there. cuML provides `cuml.model_selection.train_test_split`, `KFold`, `StratifiedKFold`, and `cuml.metrics` (`accuracy_score`, `r2_score`, `mean_squared_error`, `roc_auc_score`, `confusion_matrix`, `adjusted_rand_score`, `trustworthiness` (euclidean only), ...).
- **Output types mirror input.** cuDF in → cuDF out; CuPy in → CuPy out. Use `output_type='cupy'`, `with cuml.using_output_type('cupy'):`, or `cuml.set_global_output_type('cupy')` to force a type.
- **Numeric only.** Encode strings/categoricals on GPU first (`.astype('category').cat.codes` in cuDF, or `cuml.preprocessing` encoders).
- **Device copies still happen.** cuDF is column-major; converting a DataFrame to a dense matrix makes a GPU-side copy, so leave memory headroom (and cast to float32 to halve it).
- `.values` on a cuDF frame returns a CuPy array; `.to_cupy()` is explicit. Only call `.to_pandas()` / `.get()` / `cupy.asnumpy()` at the very end (plots, reports).

## 5. Embeddings at scale: UMAP + HDBSCAN

The standard recipe for clustering and mapping embeddings (text, image, etc.):

1. **Load as float32.**
2. **Reduce with UMAP**, typically to 5–50 dims for clustering, and separately to 2D for visualization. UMAP on GPU supports `metric='cosine'` (also euclidean, correlation, etc.), so put the embedding's natural similarity here.
3. **Cluster the UMAP output with HDBSCAN** using `metric='euclidean'`. GPU HDBSCAN only supports euclidean/l2. Don't pass `metric='cosine'` to it. If you must cluster raw embeddings with cosine semantics, L2-normalize them first so euclidean distance is monotonic in cosine distance.
4. **Validate quality, not exact match:** trustworthiness for the embedding, on a sample (it's O(n²)). `cuml.metrics.trustworthiness` only supports `metric='euclidean'`; for cosine embeddings, L2-normalize the input first (euclidean on unit vectors ranks neighbors exactly like cosine) or use `sklearn.manifold.trustworthiness(..., metric='cosine')` on the sample. For HDBSCAN: cluster count, noise fraction, and spot checks. Results differ numerically from CPU.

Typical knobs: UMAP `n_neighbors` (15–50), `min_dist` (0.0 for clustering), `n_components`. If the user needs reproducible embeddings, set `random_state` and leave `build_algo='auto'` (a seed makes `auto` use the exact brute-force kNN build, which is slower on large data; an explicit `build_algo='nn_descent'` with a seed is *not* deterministic). Conversely, setting `random_state` on a large dataset quietly gives up the faster `nn_descent` build; HDBSCAN `min_cluster_size` (scale with N), `min_samples` (`sklearn.cluster.HDBSCAN` under `cuml.accel`: 2–1024 stays on GPU, otherwise falls back; `hdbscan.HDBSCAN` and `cuml.cluster.HDBSCAN`: must be 1–1023, otherwise error).

### Sizing first

Memory ≈ rows × dims × 4 bytes (float32). 5M × 768 ≈ 15 GB; 20M × 1024 ≈ 82 GB. Leave headroom for the kNN graph and intermediates.

### When it fits on one GPU

Keep it simple. If the user already has `umap-learn` + `hdbscan` code, run it under `cuml.accel` unchanged: `umap.UMAP(metric='cosine')` and `hdbscan.HDBSCAN(metric='euclidean')` both run on GPU. Or use `cuml.UMAP` / `cuml.cluster.HDBSCAN` directly. No batching or memory tuning needed.

### When it doesn't fit on one GPU

Use the direct API's **batched kNN graph build**: the dataset is partitioned into clusters and only a fraction is on the GPU at once. Pass the data as a host NumPy array; it stays in host memory.

```python
import numpy as np
from cuml import UMAP
from cuml.cluster import HDBSCAN

X = np.load("embs.npy").astype(np.float32, copy=False)   # host memory

umap = UMAP(
    n_neighbors=30, n_components=10, min_dist=0.0, metric="cosine",
    build_algo="nn_descent",                     # or "brute_force_knn" (exact, slower)
    build_kwds={"knn_n_clusters": 8, "knn_overlap_factor": 2},
)
low = umap.fit_transform(X)
labels = HDBSCAN(min_cluster_size=500, min_samples=10).fit_predict(low)
```

- About `knn_overlap_factor / knn_n_clusters × N` rows are on the GPU at once. Start with `knn_n_clusters=4`, increase (8, 16, …) to use less GPU memory; raise `knn_overlap_factor` (2 → 3 → 4, must stay < `knn_n_clusters`) for better graph accuracy.
- Batching works with both `nn_descent` and `brute_force_knn`. Metrics for the batched build: l2/euclidean, sqeuclidean, cosine (with `nn_descent`, only l2/euclidean/sqeuclidean and cosine are supported at all).
- `cuml.cluster.HDBSCAN` takes the same `build_algo` / `build_kwds` options if you need to cluster something large directly.
- `build_kwds` can't be set through `umap.UMAP` under `cuml.accel`. The alternative there is managed memory (on by default with `cuml.accel`), which works but can be slow when heavily oversubscribed.

### Multiple GPUs on one node

`cuml.UMAP` and `cuml.cluster.HDBSCAN` accept `device_ids` (`"all"` or a list like `[0, 1, 2, 3]`) to spread the **batched kNN graph build** across GPUs. Be precise with users:
- It only takes effect with `build_algo="nn_descent"` **and** `knn_n_clusters > 1`.
- Only the kNN graph construction is multi-GPU; the rest of UMAP/HDBSCAN runs on one GPU.
- It's a direct-API option only; it isn't available under `cuml.accel`.

```python
umap = UMAP(n_components=10, metric="cosine", build_algo="nn_descent",
            build_kwds={"knn_n_clusters": 8, "knn_overlap_factor": 2},
            device_ids="all")
```

## 6. "It's barely faster": diagnosing with the profiler

Lead with measurement, then fix what it shows:

1. Profile with `n_jobs=1` on any search/CV wrapper (profilers don't see subprocesses):
   ```bash
   python -m cuml.accel --profile train.py        # per-method GPU vs CPU calls + fallback reasons
   python -m cuml.accel --line-profile train.py   # per-line time and % on GPU
   ```
   In Jupyter: `%%cuml.accel.profile` / `%%cuml.accel.line_profile`. In code: `with cuml.accel.profile(): ...`.
2. Read the report. The function profiler ends with "The following functions required CPU fallback for the following reasons", which is the to-do list. Use `accel-compatibility.md` to find alternatives.
3. Usual culprits:
   - A dominant estimator falling back (e.g. random forest with `criterion='log_loss'` or `class_weight='balanced_subsample'`, SVC on multiclass `y`, or an estimator `cuml.accel` doesn't accelerate at all).
   - Time spent in non-accelerated steps (pandas ETL → add `cudf.pandas`; unaccelerated estimators like `SimpleImputer`, `HistGradientBoosting*`).
   - Preprocessing fallbacks (scalers on scikit-learn < 1.8, `OneHotEncoder` options), or one-hot output left **sparse** so the downstream model falls back (`sparse_output=False`).
   - `n_jobs=-1` spawning many processes that contend for one GPU.
   - Ping-ponging between CPU and GPU steps (each transition copies data).
   - Small data where overhead dominates (section 7).
   - Timing that includes import time or first-call JIT. Warm up and exclude imports.
4. Fix, then **re-profile** to confirm the fallbacks are gone and compare end-to-end wall time.

## 7. When not to use a GPU

Be honest when the workload is small (roughly < 100k rows with modest features, or a run that already takes seconds). Fixed costs (CUDA context, transfers, kernel compilation) can make GPU runs slower. Recommend:
- CPU fixes first: `n_jobs=-1`, smaller grids, `HalvingGridSearchCV`/`HalvingRandomSearchCV`, fewer CV folds.
- If they still want to try, `python -m cuml.accel` on the unchanged script with a fair end-to-end timing (warm-up, imports excluded). Don't rewrite code with `cuml` imports for a small workload.
- GPUs pay off as data and algorithm cost grow: hundreds of thousands to millions of rows, kNN/UMAP/TSNE/HDBSCAN/SVM/forests, large hyperparameter searches.

## 8. Saving, loading, and deploying models

| Goal | How |
|---|---|
| Reload a direct-API cuML model on a GPU machine | `pickle.dump(model, f, protocol=5)` or `joblib.dump`. Loading requires the same cuML version. |
| Deploy a cuML-trained model to CPU-only machines (sklearn only) | `sk_model = model.as_sklearn()`, then pickle/joblib `sk_model`. Loads with only scikit-learn installed. |
| Bring an existing sklearn model to GPU | `cuml.<Estimator>.from_sklearn(sk_model)` |
| Model trained under `cuml.accel` | Pickle as usual. Loading **without** `cuml.accel` gives a normal sklearn/umap/hdbscan object; loading **with** it gives an accelerated proxy. |

`as_sklearn()`/`from_sklearn()` are available on most estimators that have sklearn counterparts (linear models, RF, KMeans, DBSCAN, HDBSCAN, PCA, TruncatedSVD, TSNE, UMAP, NearestNeighbors, SVMs, KernelRidge, encoders, ...).

Always mention:
- **Version pinning:** serialized formats aren't guaranteed compatible across cuML or dependency versions. Record the training environment and validate loading + inference in the target environment.
- **Security:** only unpickle models from trusted sources; pickle can execute arbitrary code.
