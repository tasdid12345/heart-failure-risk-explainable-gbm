Heart failure mortality: random forest baseline vs. an improved three-model ensemble.

The improved model averages class-balanced LightGBM, L2 logistic regression and random forest. Each component's hyperparameters are chosen by 5-fold stratified CV (ROC-AUC) on the training split only. The test set is used once for the final evaluation. Baseline, split, seeds (42-44) and evaluation code are unchanged.

Run: `pip install -r requirements.txt && python analysis.py`. Results are written to `results.json`, and a bar chart to `figures/auc_comparison.png`.

## Results (latest run)

On the public Heart Failure Clinical Records dataset (75-patient held-out test split), a standard random-forest baseline (scikit-learn defaults) reached ROC-AUC 0.885 (std 0.008), while a class-balanced equal-weight ensemble of LightGBM, logistic regression and random forest reached 0.899 (std 0.002), a small gain of about 0.014 with the same macro-F1 (0.803).

| Metric | Value |
|---|---|
| baseline_score | 0.8852 |
| improved_score | 0.8992 |
| improvement | 0.01403 |
| baseline_score_std | 0.007801 |
| improved_score_std | 0.002343 |
| baseline_f1 | 0.8026 |
| improved_f1 | 0.8026 |
| baseline_f1_std | 0.006308 |
| improved_f1_std | 0.006308 |
| baseline_brier | 0.1235 |
| improved_brier | 0.1222 |
| baseline_brier_std | 0.00376 |
| improved_brier_std | 0.0008359 |
| improved_cv_auc_train | 0.9265 |
| improved_lr_cv_auc_train | 0.8746 |
| improved_rf_cv_auc_train | 0.9211 |

Run on 2026-10-22. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single small dataset (299 rows, ~75 test patients, ~24 death events in test), so test AUC has wide uncertainty; seed std only captures model randomness, not split variability. The paper's own model is unspecified, so the baseline is a generic stand-in. Attributions were not computed in this script. The ensemble was not calibrated post hoc; the test set was used only for the final evaluation.

## Reference

This project is motivated by the following paper:

> RI Man, I Anghel (2023). *A machine learning based platform for remote management of heart failure patients*. 2023 IEEE 19th International Conference on. https://ieeexplore.ieee.org/abstract/document/10398604/

Paper author: Prof. Ionut Anghel, Technical University of Cluj-Napoca (ionut.anghel@cs.utcluj.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
