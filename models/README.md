# Models Directory

This directory contains trained brain age prediction models.

## Structure

```
models/
├── README.md
├── regional_bag/
│   └── atlas-4S456Parcels/
│       ├── gm_volume/
│       ├── cortical_thickness/
│       └── ...
└── evaluation/
    └── cv_results.json
```

## Model Types

- Linear (Ridge regression)
- XGBoost
- LightGBM
- MLP (neural network)

See documentation for training procedures.
