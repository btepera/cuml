#!/usr/bin/env python
"""Compare constructor parameters of direct cuML estimators with their CPU counterparts.

Prints, for each estimator, the parameters that exist only in the CPU library
(passing them to the cuML class raises TypeError) and those only in cuML.

Usage:
    python compare_params.py                  # all known pairs
    python compare_params.py RandomForestClassifier UMAP   # selected estimators

Requires cuml plus scikit-learn (and umap-learn / hdbscan for those pairs) to be
importable. Pairs whose libraries are missing are skipped.
"""

import importlib
import inspect
import sys

PAIRS = {
    "RandomForestClassifier": ("cuml.ensemble", "sklearn.ensemble"),
    "RandomForestRegressor": ("cuml.ensemble", "sklearn.ensemble"),
    "IsolationForest": ("cuml.ensemble", "sklearn.ensemble"),
    "LogisticRegression": ("cuml.linear_model", "sklearn.linear_model"),
    "LinearRegression": ("cuml.linear_model", "sklearn.linear_model"),
    "Ridge": ("cuml.linear_model", "sklearn.linear_model"),
    "Lasso": ("cuml.linear_model", "sklearn.linear_model"),
    "ElasticNet": ("cuml.linear_model", "sklearn.linear_model"),
    "KMeans": ("cuml.cluster", "sklearn.cluster"),
    "DBSCAN": ("cuml.cluster", "sklearn.cluster"),
    "HDBSCAN": ("cuml.cluster", "hdbscan"),
    "AgglomerativeClustering": ("cuml.cluster", "sklearn.cluster"),
    "PCA": ("cuml.decomposition", "sklearn.decomposition"),
    "TruncatedSVD": ("cuml.decomposition", "sklearn.decomposition"),
    "IncrementalPCA": ("cuml.decomposition", "sklearn.decomposition"),
    "TSNE": ("cuml.manifold", "sklearn.manifold"),
    "UMAP": ("cuml.manifold", "umap"),
    "NearestNeighbors": ("cuml.neighbors", "sklearn.neighbors"),
    "KNeighborsClassifier": ("cuml.neighbors", "sklearn.neighbors"),
    "KNeighborsRegressor": ("cuml.neighbors", "sklearn.neighbors"),
    "KernelDensity": ("cuml.neighbors", "sklearn.neighbors"),
    "SVC": ("cuml.svm", "sklearn.svm"),
    "SVR": ("cuml.svm", "sklearn.svm"),
    "LinearSVC": ("cuml.svm", "sklearn.svm"),
    "LinearSVR": ("cuml.svm", "sklearn.svm"),
    "KernelRidge": ("cuml.kernel_ridge", "sklearn.kernel_ridge"),
}


def params(module, name):
    cls = getattr(importlib.import_module(module), name)
    return {
        p
        for p, v in inspect.signature(cls.__init__).parameters.items()
        if p != "self" and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)
    }


def main(names):
    for name in names or PAIRS:
        if name not in PAIRS:
            print(f"## {name}: unknown (known: {', '.join(PAIRS)})")
            continue
        gpu_mod, cpu_mod = PAIRS[name]
        try:
            gpu, cpu = params(gpu_mod, name), params(cpu_mod, name)
        except (ImportError, AttributeError) as e:
            print(f"## {name}: skipped ({e})")
            continue
        print(f"## {name}  ({gpu_mod} vs {cpu_mod})")
        print(f"  CPU-only (not accepted by cuML): {sorted(cpu - gpu) or '-'}")
        print(f"  cuML-only:                       {sorted(gpu - cpu) or '-'}")


if __name__ == "__main__":
    main(sys.argv[1:])
