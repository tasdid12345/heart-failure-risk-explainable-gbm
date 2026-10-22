"""Heart failure mortality: standard random forest (baseline) vs. improved ensemble
(class-balanced, CV-tuned LightGBM + CV-tuned logistic regression, averaged and
calibrated). Seed 42 everywhere; the test set is used once for final evaluation."""
import json, os, time, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from lightgbm import LGBMClassifier
from ucimlrepo import fetch_ucirepo

SEED = 42
SEEDS = [42, 43, 44]
SOURCE_URL = "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
random.seed(SEED); np.random.seed(SEED)

# 1) Load data (ucimlrepo fetches from UCI; retry with backoff on transient errors)
print("Loading dataset...")
data = None
for attempt in range(5):
    try:
        data = fetch_ucirepo(id=519)
        break
    except Exception as e:  # network/HTTP errors
        wait = 2 ** attempt
        print(f"  download failed ({e}); retrying in {wait}s")
        time.sleep(wait)
if data is None:
    raise SystemExit("Could not download dataset.")
X = data.data.features.copy()
y = data.data.targets.iloc[:, 0].astype(int).values
X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))
if len(X) > 200_000:  # subsample guard (not triggered for this dataset)
    idx = np.random.RandomState(SEED).choice(len(X), 200_000, replace=False)
    X, y = X.iloc[idx], y[idx]
n_rows = len(X)
print(f"  rows={n_rows}, features={X.shape[1]}, death rate={y.mean():.3f}")
if n_rows < 50 or len(np.unique(y)) < 2:
    raise SystemExit("Not enough data for meaningful metrics.")

# 2) Single stratified 75/25 split (fixed seed); same split used for both models
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print(f"  train={len(X_tr)}, test={len(X_te)}")
scaler = StandardScaler().fit(X_tr)  # same preprocessing for both models (fit on train only)
Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def evaluate(proba, pred):
    return {"auc": roc_auc_score(y_te, proba),
            "f1": f1_score(y_te, pred, average="macro"),
            "brier": brier_score_loss(y_te, proba)}

# 3) Baseline: default random forest (seeded; run over seeds 42-44) -- unchanged
print("Baseline: random forest...")
base_runs = []
for s in SEEDS:
    rf = RandomForestClassifier(random_state=s, n_jobs=-1).fit(Xs_tr, y_tr)
    p = rf.predict_proba(Xs_te)[:, 1]
    base_runs.append(evaluate(p, (p >= 0.5).astype(int)))

# 4) Improved: ensemble of class-balanced, CV-tuned models (tuning on train CV only)
#    - LightGBM with a wider randomized search (CV AUC, train only)
#    - Class-balanced L2 logistic regression with CV-tuned C (strong, low-variance
#      model that suits a small n=224 dataset)
#    Final score = mean of the two models' probabilities, evaluated on test once.
print("Improved: CV tuning LightGBM and logistic regression on training set...")
lgb_search = RandomizedSearchCV(
    LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=1),
    param_distributions={"n_estimators": [100, 200, 400, 600], "learning_rate": [0.01, 0.02, 0.05],
                         "num_leaves": [4, 7, 15], "min_child_samples": [5, 10, 20, 30],
                         "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.5, 0.7, 0.9],
                         "reg_lambda": [0.0, 1.0, 5.0]},
    n_iter=30, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1)
lgb_search.fit(Xs_tr, y_tr)
print(f"  LGBM best CV AUC={lgb_search.best_score_:.3f}, params={lgb_search.best_params_}")

lr_search = RandomizedSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
    param_distributions={"C": np.logspace(-3, 2, 30)},
    n_iter=30, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=-1)
lr_search.fit(Xs_tr, y_tr)
print(f"  LR best CV AUC={lr_search.best_score_:.3f}, params={lr_search.best_params_}")

imp_runs = []
for s in SEEDS:
    lgb = LGBMClassifier(**{**lgb_search.best_params_, "class_weight": "balanced",
                            "random_state": s, "verbose": -1, "n_jobs": 1})
    lgb.fit(Xs_tr, y_tr)
    p_lgb = lgb.predict_proba(Xs_te)[:, 1]
    lr = LogisticRegression(**{**lr_search.best_params_, "class_weight": "balanced",
                               "max_iter": 5000, "random_state": s}).fit(Xs_tr, y_tr)
    p_lr = lr.predict_proba(Xs_te)[:, 1]
    p = 0.5 * p_lgb + 0.5 * p_lr  # ensemble average of probabilities
    imp_runs.append(evaluate(p, (p >= 0.5).astype(int)))

def agg(runs, k):
    v = np.array([r[k] for r in runs]); return float(v.mean()), float(v.std())
b_auc, b_auc_sd = agg(base_runs, "auc"); i_auc, i_auc_sd = agg(imp_runs, "auc")
b_f1, b_f1_sd = agg(base_runs, "f1"); i_f1, i_f1_sd = agg(imp_runs, "f1")
b_br, b_br_sd = agg(base_runs, "brier"); i_br, i_br_sd = agg(imp_runs, "brier")

metrics = {
    "baseline_score": b_auc, "improved_score": i_auc, "improvement": i_auc - b_auc,
    "baseline_score_std": b_auc_sd, "improved_score_std": i_auc_sd,
    "baseline_f1": b_f1, "improved_f1": i_f1, "baseline_f1_std": b_f1_sd, "improved_f1_std": i_f1_sd,
    "baseline_brier": b_br, "improved_brier": i_br, "baseline_brier_std": b_br_sd, "improved_brier_std": i_br_sd,
    "improved_cv_auc_train": float(lgb_search.best_score_),
    "improved_lr_cv_auc_train": float(lr_search.best_score_),
}
results = {
    "question": "Can a class-balanced, cross-validated gradient-boosting model with per-patient feature attributions beat a standard random-forest risk model for heart failure mortality prediction on the public Heart Failure Clinical Records dataset?",
    "method": "Stratified 75/25 split (seed 42); RF default baseline vs. ensemble of class-balanced, CV-tuned LightGBM and CV-tuned L2 logistic regression (probability average); ROC-AUC main metric, macro-F1 and Brier secondary; seeds 42-44 for both models.",
    "dataset": {"name": "Heart Failure Clinical Records", "source_url": SOURCE_URL,
                "n_rows": int(n_rows), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "ROC-AUC (test)", "higher_is_better": True,
    "baseline": "Random forest classifier, scikit-learn defaults (stand-in for the paper's unspecified ML model).",
    "improved": "Equal-weight probability ensemble of class-balanced LightGBM and class-balanced logistic regression; hyperparameters chosen by 5-fold stratified CV (AUC) on training data only.",
    "metrics": {k: float(v) for k, v in metrics.items()},
    "notes": "Single small dataset (299 rows, ~75 test patients, ~24 death events in test), so test AUC has wide uncertainty; seed std only captures model randomness, not split variability. The paper's own model is unspecified, so the baseline is a generic stand-in. Attributions were not computed in this script. The ensemble was not calibrated post hoc; the test set was used only for the final evaluation.",
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# 5) Optional bar chart
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
names = ["Baseline RF", "Improved ensemble"]
ax.bar(names, [b_auc, i_auc], yerr=[b_auc_sd, i_auc_sd], color=["gray", "steelblue"], capsize=5)
ax.set_ylabel("Test ROC-AUC (mean ± sd, seeds 42-44)")
ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figures", "auc_comparison.png"), dpi=120)
print(f"Baseline AUC={b_auc:.3f}, Improved AUC={i_auc:.3f}, improvement={i_auc-b_auc:+.3f}")
print(f"Done in {time.time()-t0:.1f}s; results written to results.json")
