# Term Deposit Call Targeting

Two-layer machine learning system that raises the success rate of a bank's term-deposit call campaign and cuts wasted agent time, while staying interpretable enough for the client to act on.

## Problem

A European bank calls customers to sell term deposits. Only about 7% subscribe, so calling everyone burns most of the agents' time on people who will never say yes. The goal is to concentrate calls on the customers most likely to convert.

## Data

40,000 call records. `term-deposit-marketing-2020.csv`.

- Numeric: age, balance, duration, campaign
- Categorical: job, marital, education, default, housing, loan, contact, month
- Target `y`: subscribed yes/no, imbalanced at roughly 93% no / 7% yes

## Approach

The pipeline runs in two layers.

**Layer 1, filter.** XGBoost predicts who will not subscribe and removes them from the call list. It is tuned for recall on the yes class: we accept calling some non-subscribers to avoid dropping real ones. Class imbalance is handled with resampling (RandomUnderSampler, SMOTETomek, SMOTEENN tested) and `scale_pos_weight`, with hyperparameters tuned by Hyperopt. Logistic Regression and Random Forest are used as baselines.

**Layer 2, rank.** A second XGBoost scores everyone Layer 1 kept and sorts them by subscription probability, so agents dial the strongest leads first. This turns a yes/no filter into a priority queue.

After the supervised pipeline, KMeans segments the customers for interpretability. Because categorical features cannot be averaged, cluster profiles combine numeric means with the mode (most frequent category) per feature. Clusters are visualized with t-SNE, PCA, and UMAP.

## Results

Layer 1 (test set, 8,000 held out):

- Calls skipped: 3,815 of 8,000 (48%)
- Subscribers retained: 73%
- Agent-hours saved: about 778

Layer 2 (4,185 kept, 421 subscribers):

| Catch this share of subscribers | Calls to make | Share of pool |
|---|---|---|
| 80% | 735 | 18% |
| 90% | 1,060 | 25% |
| 100% | 3,184 | 76% |

Combined: about 1,000 agent-hours saved end to end while keeping most conversions.

### Segmentation findings

- Callback clusters (k=4): loan status is the hidden signal. Debt-free, higher-balance customers convert around 12 to 14%; customers carrying a housing or personal loan convert around 8%. Existing debt predicts lower deposit uptake.
- Full dataset (k=3): splits mainly on age. Older (about 57) and younger (about 34) segments convert around 10 to 11%; the mid-age majority (about 69% of the base) converts around 6%.
- t-SNE, PCA, and UMAP agree: one genuinely distinct high-conversion segment (older customers) plus two large groups split along a fuzzy age and response gradient.

## Success metric

Target: 81% or higher accuracy under 5-fold cross-validation, averaged.

Random Forest and Decision Tree clear 81% accuracy, but accuracy is misleading here. The base rate is about 93% no, so a model that predicts no for everyone scores about 0.93 and catches zero subscribers. Random Forest does roughly this: 0.92 accuracy but only 0.07 recall on the yes class. The metric is met on paper but the models that meet it are not useful.

The design is instead optimized for recall and F1 on the yes class, which is what call targeting actually needs. Report the 5-fold accuracy to satisfy the requirement, then evaluate on yes-class recall for the real decision.

## Limitations

- Cluster silhouettes are low (about 0.13 on the full set, about 0.25 on callbacks). Treat clusters as response tiers along a continuum, not distinct personas. The older high-conversion segment is the one clean exception.
- KMeans on mixed one-hot and numeric features with Euclidean distance is theoretically weak. k-prototypes would be the correct method for sharper mixed-type segments.
- `duration` is only known after a call ends, so it leaks and cannot be used for pre-call targeting in production. It must be removed before deployment, and the time-saved figures partly depend on it.

## Running it

Open `Project_2.ipynb` and run top to bottom. The unsupervised and visualization cells depend on variables built earlier in the notebook (`X_test`, `models`, `df_encoded`, `Xs_all`), so run in order.

### Dependencies

```
pandas numpy scipy matplotlib seaborn missingno
scikit-learn xgboost hyperopt imbalanced-learn
duckdb umap-learn
```

UMAP installs as `umap-learn` and imports as `umap`.

## Files

- `Project_2.ipynb` — full pipeline: EDA, statistical tests, two-layer modeling, business impact, clustering, dimensionality reduction
- `term-deposit-marketing-2020.csv` — dataset

## Apziva

Project code: 4MSpWOdDLp519rH3
