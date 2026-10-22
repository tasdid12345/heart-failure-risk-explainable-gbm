Heart failure mortality: random forest baseline vs. an improved three-model ensemble.

The improved model averages class-balanced LightGBM, L2 logistic regression and random forest. Each component's hyperparameters are chosen by 5-fold stratified CV (ROC-AUC) on the training split only. The test set is used once for the final evaluation. Baseline, split, seeds (42-44) and evaluation code are unchanged.

Run: `pip install -r requirements.txt && python analysis.py`. Results are written to `results.json`, and a bar chart to `figures/auc_comparison.png`.

## Reference

This project is motivated by the following paper:

> RI Man, I Anghel (2023). *A machine learning based platform for remote management of heart failure patients*. 2023 IEEE 19th International Conference on. https://ieeexplore.ieee.org/abstract/document/10398604/

Paper author: Prof. Ionut Anghel, Technical University of Cluj-Napoca (ionut.anghel@cs.utcluj.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
