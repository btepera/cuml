# cuml.accel: accelerated estimators and CPU-fallback conditions

Condensed from `docs/source/cuml-accel/compatibility.rst` (cuML 26.12) and **corrected against the source code** (the `_params_from_cpu` / `UnsupportedOnGPU` checks in `python/cuml/cuml/**`), which disagrees with that doc in several places. When the two conflict, trust this file or the code.

**How fallback works:** an estimator falls back only when the GPU path raises cuML's internal `UnsupportedOnGPU`. Any *other* error on the GPU path (e.g. an invalid metric for sparse input) surfaces to the user as an exception rather than silently falling back. Configuring sklearn callbacks via `set_callbacks` forces CPU fallback for any accelerated estimator.

**Tested versions:** scikit-learn 1.6–1.9.1, umap-learn 0.5.7–0.5.12, hdbscan 0.8.39–0.8.44 (others warn but usually work). Some estimators (the scalers) use sklearn's Array API support and need **scikit-learn ≥ 1.8** for GPU execution.

Anything **not listed** here runs on CPU unchanged (e.g. `HistGradientBoosting*`, `GradientBoosting*`, `ExtraTrees*`, `SimpleImputer`, `CountVectorizer`/`TfidfVectorizer`, `MiniBatchKMeans`, `Birch`, `NMF`, `LDA`, `GaussianMixture`, …). Only estimator classes are intercepted, not standalone functions.

"Sparse X" below means scipy/cupyx sparse input.

## sklearn.cluster
| Estimator | Falls back to CPU when |
|---|---|
| `KMeans` | callable `init`; sparse X |
| `DBSCAN` | `algorithm` not `auto`/`brute`; `metric` not in l2, euclidean, cosine, precomputed; sparse X |
| `SpectralClustering` | `assign_labels != "kmeans"`; `affinity` not `nearest_neighbors`/`precomputed` (**sklearn default `rbf` falls back**); sparse X. `affinity_matrix_` not computed |
| `HDBSCAN` (sklearn) | `metric` not l2/euclidean; non-empty `metric_params`; `store_centers` set; effective `min_samples` outside 2–1024 (incl. `min_samples=1`; sklearn counts the point itself, so this range is sklearn-specific); sparse / precomputed / non-finite input |

## sklearn.covariance
`EmpiricalCovariance`, `LedoitWolf`: no estimator-specific limitations.

## sklearn.decomposition
| Estimator | Falls back to CPU when |
|---|---|
| `PCA` | `n_components=0` or `"mle"`. Any `svd_solver` runs on GPU (mapped to cuML's own solvers); randomized-solver params (`random_state`, `n_oversamples`, `power_iteration_normalizer`) ignored. A float `n_components` (variance ratio) does **not** fall back and isn't handled like sklearn (upstream tests are xfailed), so prefer an integer |
| `IncrementalPCA` | none (`partial_fit` doesn't take sparse, same as sklearn) |
| `TruncatedSVD` | sparse X. Randomized-solver params ignored |

## sklearn.ensemble
| Estimator | Falls back to CPU when |
|---|---|
| `RandomForestClassifier` | `criterion="log_loss"`; callable `oob_score`; `warm_start=True`; `monotonic_cst`; integer `max_samples`; `min_weight_fraction_leaf != 0`; `ccp_alpha != 0`; `class_weight="balanced_subsample"`; sparse X; NaN in X; multi-output y. **Runs on GPU:** `class_weight` dict or `"balanced"`, and `sample_weight` in fit/score (the upstream doc wrongly lists these as fallbacks) |
| `RandomForestRegressor` | `criterion` `absolute_error`/`friedman_mse`; callable `oob_score`; `warm_start=True`; `monotonic_cst`; integer `max_samples`; `min_weight_fraction_leaf != 0`; `ccp_alpha != 0`; sparse X; NaN in X; multi-output y. `sample_weight` runs on GPU |
| `IsolationForest` | `warm_start=True`; `sample_weight` passed; sparse X; NaN/inf in X. `estimators_samples_` not computed |

## sklearn.kernel_ridge
`KernelRidge`: sparse X; non-string `kernel`. Results ≈ identical to sklearn.

## sklearn.linear_model
| Estimator | Falls back to CPU when |
|---|---|
| `LinearRegression` | `positive=True`. `rank_`, `singular_` not computed |
| `LogisticRegression` | `warm_start=True`; `intercept_scaling != 1`; `multi_class` set to anything other than `"auto"`/`"deprecated"`. Any sklearn `solver` runs on GPU (cuML always uses its own `qn` solver). `n_iter_` is computed |
| `ElasticNet`, `Lasso` | `positive=True`; `warm_start=True`; `precompute` not False; multi-output y. `dual_gap_` not computed |
| `Ridge` | `positive=True` or `solver="lbfgs"`; sparse X with `solver` in svd/cholesky/sag/saga |

## sklearn.manifold
| Estimator | Falls back to CPU when |
|---|---|
| `TSNE` | `n_components != 2`; array `init`; `init="pca"` with sparse X (checked in `fit` only); `metric` not in l2, euclidean, sqeuclidean, cityblock, l1, manhattan, minkowski, chebyshev, cosine, correlation. Not fully deterministic |
| `SpectralEmbedding` | `affinity` not `nearest_neighbors`/`precomputed`; sparse X; single feature. `affinity_matrix_` not computed |

## sklearn.neighbors
| Estimator | Falls back to CPU when |
|---|---|
| `NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor` | `metric` not in l2, euclidean, l1, cityblock, manhattan, taxicab, canberra, minkowski, lp, chebyshev, linf, jensenshannon, cosine, correlation, inner_product, sqeuclidean, haversine; callable `weights` (KNeighbors*). sklearn `algorithm="auto"` lets cuML choose (random ball cover for 2–3 dense features with euclidean/haversine, else brute force); other values → brute force. `radius_neighbors` always CPU; `radius_neighbors_graph` runs on GPU for `mode="connectivity"`, `sort_results=False`, dense, l2/euclidean. Sparse X with jensenshannon/correlation/sqeuclidean/haversine **raises an error** instead of falling back |
| `KernelDensity` | `metric` not in cityblock, cosine, euclidean, l1, l2, manhattan, sqeuclidean, canberra, chebyshev, minkowski, hellinger, correlation, jensenshannon, hamming, kldivergence, russellrao. (`nan_euclidean` passes the check but then **errors** on GPU.) `algorithm`, `atol`, `rtol`, `breadth_first`, `leaf_size` ignored |

## sklearn.preprocessing
| Estimator | Falls back to CPU when |
|---|---|
| `StandardScaler` | sparse X; **scikit-learn < 1.8** |
| `MinMaxScaler` | sparse X; scikit-learn < 1.8 |
| `MaxAbsScaler` | sparse X; scikit-learn < 1.8 |
| `PolynomialFeatures` | sparse X; `order="F"`; scikit-learn < 1.8 |
| `LabelEncoder`, `LabelBinarizer` | none |
| `OneHotEncoder` | `bytes` values; explicit `categories` with mixed-type objects; `dtype` not float/bool; `drop="if_binary"`; `handle_unknown` `warn`/`infrequent_if_exist`; `min_frequency` set; `max_categories` set; callable `feature_name_combiner`. Treats None and NaN as the same category |
| `TargetEncoder` | `categories != "auto"`; target that isn't binary or continuous (multiclass, multilabel, multi-output); `random_state` that isn't `None` or a Python `int` (NumPy ints and `RandomState` fall back). Different CV folds in `fit_transform` than sklearn |

## sklearn.svm
| Estimator | Falls back to CPU when |
|---|---|
| `SVC` | callable `kernel`; **multiclass y**; **`probability=True`** (see the SVC row in `accelerating-workflows.md` for alternatives) |
| `SVR` | callable `kernel` |

`kernel="precomputed"` runs on GPU for both SVC and SVR (the upstream doc wrongly lists it as a fallback); sparse precomputed kernels raise an error, as in sklearn.
| `LinearSVC` | sparse X; `intercept_scaling != 1`; `multi_class != "ovr"` |
| `LinearSVR` | sparse X; `intercept_scaling != 1` |

## umap.UMAP
Falls back when: string `init` other than `random`/`spectral` (array `init` runs on GPU); `metric` not in l1, cityblock, taxicab, manhattan, euclidean, l2, sqeuclidean, canberra, minkowski, chebyshev, linf, cosine, correlation, hellinger, hamming (`jaccard` is listed upstream but always falls back); `target_metric` not categorical/l2/euclidean; `unique=True`; `densmap=True`; `ensure_all_finite` not True. `output_metric` is **silently ignored** on GPU (output is always euclidean). Setting `random_state` makes results reproducible: with the default `build_algo='auto'`, a seed forces the exact brute-force kNN build (otherwise `auto` picks `nn_descent` above 50k rows). The cost is speed on large data. Only an explicit `build_algo='nn_descent'` plus `random_state` is non-deterministic (cuML warns). Recommend `numba<0.62` for stability.

## hdbscan.HDBSCAN
Falls back when: `metric` not l2/euclidean; `memory` set; `match_reference_implementation=True`; `branch_detection_data=True`; sparse X. `min_samples` must be 1–1023 (this library doesn't count the point itself); values outside that range raise an error rather than falling back. Don't mix this up with the sklearn `HDBSCAN` range above. Not computed: `exemplars_`, `outlier_scores_`, `relative_validity_`. Non-deterministic with duplicate points.

## Not supported via ONNX export (skl2onnx)
DBSCAN, TSNE, SpectralEmbedding, NearestNeighbors, UMAP, HDBSCAN.
