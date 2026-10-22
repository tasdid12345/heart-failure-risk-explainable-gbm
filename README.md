# Heart Failure Mortality: Random Forest vs. Calibrated Gradient Boosting

**Motivation.** Remote management platforms for heart failure patients increasingly rely on machine learning to flag high-risk patients between clinic visits. The paper "A machine learning based platform for remote management of heart failure patients" (2023) describes such a web-based system but does not state which algorithm it uses, so a transparent, reproducible comparison of standard risk models on public clinical data is useful.

**Question.** Can a class-balanced, cross-validated gradient-boosting model with per-patient feature attributions beat a standard random-forest risk model for heart failure mortality prediction on the public Heart Failure Clinical Records dataset?

**Dataset.** Heart Failure Clinical Records (UCI, id 519), 299 patients with 13 clinical features and a death-event label, loaded via the `ucimlrepo` package. This is public data and is not the paper's own dataset.

**Baseline vs. improved.**
- Baseline: random forest classifier with scikit-learn default settings (a stand-in for the paper's unspecified model).
- Improved: LightGBM with balanced class weights, hyperparameters selected by stratified 5-fold CV on the training split only, and sigmoid probability calibration.
- Both use the same stratified 75/25 split (seed 42), the same standard scaling, and the same test set. The main metric is ROC-AUC; macro-F1 and Brier score are secondary.

**How to run**
```
pip install -r requirements.txt
python analysis.py
```

**Results** are written to `results.json` (and `figures/auc_comparison.png`).

## Reference

This project is motivated by the following paper:

> RI Man, I Anghel (2023). *A machine learning based platform for remote management of heart failure patients*. 2023 IEEE 19th International Conference on. https://ieeexplore.ieee.org/abstract/document/10398604/

Paper author: Prof. Ionut Anghel, Technical University of Cluj-Napoca (ionut.anghel@cs.utcluj.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
