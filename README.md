
# E-Commerce Return Risk Prediction & Recommendation System

Predicts the likelihood that an e-commerce order will be returned, and — when risk is high —
recommends safer alternative products, using a content-based recommendation engine.

## Why this project exists

Returns are one of the largest hidden costs in e-commerce, particularly in fashion, where
size/fit mismatches drive the majority of returns. This project builds an end-to-end system
that flags high-return-risk orders *before* they ship, so a business could intervene early
(better size guidance, or suggesting a safer alternative product) rather than absorbing the
cost of reverse logistics after the fact.

## Architecture

```
React frontend (form)
       │
       │  POST /predict  { raw order details }
       ▼
FastAPI backend
       │
       ├── reconstructs 60 engineered/encoded features from raw input
       ├── XGBoost model → return_probability
       │
       │  if risk is Medium/High:
       ▼
POST /recommendations  { category, fabric, fit_type, price }
       │
       ├── maps to a "virtual product" group (see note below)
       ├── content-based filtering (cosine similarity) within same category
       ├── excludes alternatives with equal/higher return rate
       └── returns safer alternatives, backed by real sample sizes (≥10 orders)
```

**Note on "virtual products":** the dataset is order-level, not catalog-level — there's no
`product_id`, so no product is ever purchased more than once under a shared identifier. To make
recommendations statistically meaningful (not based on a single order's outcome), products are
grouped into "virtual products" by shared attributes (category + fabric + fit_type +
price_bucket), giving each group real aggregate return-rate statistics.

## Dataset

~50,000 fashion e-commerce orders, 24 raw features, ~28% return rate. Source:
[Kaggle — Will This Order Come Back?](https://www.kaggle.com/datasets/sergionefedov/will-this-order-come-back)

**Important:** the raw `return_reason` column directly encodes the target (it's `not_returned`
exactly when `returned=0`) and is dropped before modeling to avoid data leakage.

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | (baseline) |
| Random Forest | (comparison) |
| **XGBoost (winner)** | **0.8421** |

- Precision: 0.5699
- Recall: 0.7291
- F1-score: 0.6397

**Top predictive features:** `size_mismatch`, `mismatch_direction`, `customer_prior_return_rate`,
`category_accessories`, `ordered_multiple_sizes`

Recall was prioritized over precision via class-imbalance handling (`scale_pos_weight` in
XGBoost) — in this business context, missing an actual return (false negative) is costlier than
an unnecessary alert (false positive).

## Tech stack

**ML/Data:** Python, Pandas, Scikit-learn, XGBoost
**Backend:** FastAPI, Pydantic, Joblib
**Frontend:** React (Vite)

## Project structure

```
├── data/                    # raw + processed datasets
├── notebooks/                # data exploration, feature engineering, training, recommendations
│   └── outputs/               # EDA plots, feature importance chart
├── backend/
│   ├── app/                  # FastAPI application (main, model, recommend, preprocessing, schemas)
│   └── models/                # trained model + similarity artifacts (.pkl)
└── frontend/
    └── src/                   # React app (form + results)
```

## Running locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs available at `http://localhost:8000/docs`

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```
App available at `http://localhost:5173`

## Rebuilding the model from scratch

```bash
python notebooks/01_explore.py
python notebooks/02_feature_engineering.py
python notebooks/03_train_model.py
python notebooks/04_recommendation_engine.py
```

## Key design decisions

- **No data leakage:** `return_reason` is dropped entirely before modeling since it encodes the
  outcome directly; engineered features avoid using any information not available before shipping.
- **Imbalance-aware evaluation:** with a ~28% positive class, accuracy is misleading — models are
  compared using Precision, Recall, F1, and ROC-AUC instead.
- **Statistically grounded recommendations:** alternative products are only suggested if backed
  by at least 10 historical orders, avoiding recommendations based on noisy, small-sample return
  rates.
- **Training-serving consistency:** the backend reconstructs features (including price-bucket
  quantile boundaries) using the exact same logic and boundaries used during training, avoiding
  training/serving skew.

## Live demo

- Frontend: _add your Vercel URL here_
- Backend API docs: _add your Render URL + /docs here_
```
