# Direct cuML API vs sklearn / umap-learn / hdbscan: constructor differences

Use this before writing or fixing any `cuml.*` estimator constructor. Passing a CPU-only parameter to a cuML class raises `TypeError: __init__() got an unexpected keyword argument`. If cuML is installed in your environment, you can run the skill's `scripts/compare_params.py [Estimator ...]` to print the live comparison (it's bundled with the skill, so don't tell the user to run it; give them `inspect.signature(cuml.X)` instead).

Generated from cuML 26.12 source vs scikit-learn 1.9.1, umap-learn 0.5.12, hdbscan 0.8.x. `output_type` and `verbose` are cuML-only on every estimator and omitted below.

## Parameters the CPU class has but cuML does not

| cuML estimator | CPU-only parameters (remove or translate) | Notable cuML-only parameters |
|---|---|---|
| `cuml.ensemble.RandomForestClassifier` / `RandomForestRegressor` | `criterion` (→ **`split_criterion`**), `n_jobs`, `warm_start`, `ccp_alpha`, `min_weight_fraction_leaf`, `max_leaf_nodes` (→ `max_leaves`), `monotonic_cst`. (The classifier **does** accept `class_weight`: dict or `'balanced'`, not `'balanced_subsample'`.) | `split_criterion`, `max_leaves`, `n_bins`, `n_streams`, `max_batch_size` |
| `cuml.ensemble.IsolationForest` | `n_jobs`, `warm_start` | `max_depth` |
| `cuml.linear_model.LogisticRegression` | `n_jobs`, `warm_start`, `dual`, `intercept_scaling`, `random_state` | `lbfgs_memory`, `linesearch_max_iter`, `penalty_normalized` |
| `cuml.linear_model.LinearRegression` | `n_jobs`, `positive`, `tol` | `algorithm` |
| `cuml.linear_model.Ridge` | `positive`, `random_state` | — |
| `cuml.linear_model.Lasso` / `ElasticNet` | `copy_X`, `positive`, `precompute`, `random_state`, `warm_start` | `solver` |
| `cuml.cluster.KMeans` | `algorithm`, `copy_x` | `oversampling_factor`, `max_samples_per_batch`, `init_size` |
| `cuml.cluster.DBSCAN` | `n_jobs`, `leaf_size`, `metric_params`, `p` | `calc_core_sample_indices`, `max_mbytes_per_batch` |
| `cuml.cluster.HDBSCAN` (vs `hdbscan.HDBSCAN`) | `algorithm`, `leaf_size`, `core_dist_n_jobs`, `approx_min_span_tree`, `memory`, `match_reference_implementation`, `branch_detection_data`, `cluster_selection_epsilon_max`, `cluster_selection_persistence`, `**kwargs` | `build_algo`, `build_kwds`, `device_ids` |
| `cuml.cluster.AgglomerativeClustering` | `distance_threshold`, `compute_full_tree`, `memory` | `c` |
| `cuml.decomposition.PCA` | `random_state`, `n_oversamples`, `power_iteration_normalizer` | — |
| `cuml.decomposition.TruncatedSVD` | `n_oversamples`, `power_iteration_normalizer` | — |
| `cuml.decomposition.IncrementalPCA` | — | — |
| `cuml.manifold.TSNE` | `n_jobs` | `n_neighbors`, `learning_rate_method`, `exaggeration_iter`, `late_exaggeration`, `pre_momentum`, `post_momentum`, `precomputed_knn` |
| `cuml.UMAP` (vs `umap.UMAP`) | `n_jobs`, `low_memory`, `unique`, `densmap` (+ `dens_*`, `output_dens`), `output_metric`, `output_metric_kwds`, `target_metric_kwds`, `transform_seed`, `transform_mode`, `angular_rp_forest`, `force_approximation_algorithm`, `disconnection_distance`, `tqdm_kwds` | `build_algo`, `build_kwds`, `device_ids`, `hash_input`, `callback` |
| `cuml.neighbors.NearestNeighbors` / `KNeighborsClassifier` / `KNeighborsRegressor` | `leaf_size` (`n_jobs` is accepted but ignored) | `algo_params` |
| `cuml.neighbors.KernelDensity` | `algorithm`, `atol`, `rtol`, `breadth_first`, `leaf_size` | — |
| `cuml.svm.SVC` | `probability`, `break_ties`, `shrinking` | `nochange_steps` |
| `cuml.svm.SVR` | `shrinking` | `nochange_steps` |
| `cuml.svm.LinearSVC` / `LinearSVR` | `dual`, `intercept_scaling`, `random_state` | `penalized_intercept`, `lbfgs_memory`, `linesearch_max_iter` |
| `cuml.kernel_ridge.KernelRidge` | — | — |

## Parameters that exist in both but accept fewer values in cuML

| cuML estimator | Parameter | cuML accepts |
|---|---|---|
| `LogisticRegression` | `solver` | `'qn'` only (auto-selects L-BFGS or OWL-QN; L1/elasticnet penalties still work). Drop `'saga'`, `'liblinear'`, `'lbfgs'`, … |
| `DBSCAN` | `algorithm` | `'brute'`, `'rbc'` (not `'ball_tree'`, `'kd_tree'`) |
| `HDBSCAN` | `metric` | `'euclidean'` only. For cosine semantics, reduce with UMAP (`metric='cosine'`) first or L2-normalize |
| `HDBSCAN` | `min_samples` | 1–1023 (raises otherwise) |
| `PCA` | `svd_solver` | `'auto'`, `'full'` (default), `'jacobi'`; `'randomized'`, `'arpack'`, `'covariance_eigh'` raise `ValueError` |
| `UMAP` | `metric` | dense: l1/cityblock/manhattan/taxicab, euclidean/l2, sqeuclidean, canberra, minkowski, chebyshev/linf, cosine, correlation, hellinger, hamming (not `inner_product`); with `build_algo='nn_descent'`: euclidean/l2, sqeuclidean, cosine only |
| `NearestNeighbors` & friends | `algorithm` | `'auto'`, `'brute'`, `'rbc'` (2D euclidean/haversine), `'ivfflat'`, `'ivfpq'` (approximate) |
| `KMeans` | `init` | `'scalable-k-means++'`/`'k-means||'` (default), `'k-means++'`, `'random'`, or an array |
| `RandomForest*` | `split_criterion` | classifier: `'gini'` (default), `'entropy'`; regressor: `'mse'` (default), `'poisson'`, `'gamma'`, `'inverse_gaussian'`. **Use `'mse'`, not sklearn's `'squared_error'`.** An unrecognized value only emits a warning and continues with an invalid criterion; it doesn't raise. No `'log_loss'`, `'absolute_error'`, `'friedman_mse'` |
| `UMAP` | `build_algo` | `'auto'` (brute force ≤ 50k rows, else nn_descent), `'brute_force_knn'`, `'nn_descent'` |
| `UMAP` / `HDBSCAN` | `build_kwds` keys | `knn_n_clusters`, `knn_overlap_factor`, `nnd_graph_degree`, `nnd_intermediate_graph_degree`, `nnd_max_iterations`, `nnd_termination_threshold`. Unknown keys (e.g. `nnd_n_clusters`, a typo in some docstrings) are **silently ignored**. `fit`/`fit_transform` take no `data_on_host` argument; host-resident input is handled automatically |
| `TSNE` | `n_components` | `2` only |

## Rule of thumb

- Don't pass `n_jobs` to cuML estimators; only the neighbors estimators accept it, and they ignore it.
- If the user wants to keep their sklearn constructor calls exactly as they are, use `cuml.accel` instead of the direct API. It accepts the full sklearn signature and falls back to CPU for unsupported values.
- If unsure whether a parameter exists, say so and check (the skill's `compare_params.py`, or tell the user to run `help(cuml.X)` / `inspect.signature(cuml.X)`) rather than guessing.
