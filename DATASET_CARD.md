# Dataset Card: Synthetic Serving Fixture

The training script generates 5,000 binary classification rows with `sklearn.datasets.make_classification`, eight numeric features, six informative dimensions, two redundant dimensions, 70/30 class weights, class separation 1.2, and seed 42.

The data is synthetic, generated locally, contains no personal information, and is not presented as scientific or production data. It is used because this repository evaluates model lifecycle and serving mechanics rather than domain accuracy. A real deployment would require a separately governed dataset, domain validation, drift analysis, and approval criteria.
