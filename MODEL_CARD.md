# Model Card

The artifact is a scikit-learn pipeline containing `StandardScaler` and logistic regression. Version `1.0.0` expects exactly eight finite numeric values and returns a binary class plus positive-class probability.

It is intended only as a reproducible serving fixture. It is not calibrated for any real domain, has no fairness or drift validation, and must not be used for consequential decisions. Training metadata and measured test metrics are stored in `artifacts/metadata.json`.

The executed seed-42 run trained on 4,000 synthetic rows and measured 0.8390 accuracy and 0.7190 F1 on 1,000 held-out rows. These metrics validate artifact consistency for the serving workflow; they do not describe a real-world task.
