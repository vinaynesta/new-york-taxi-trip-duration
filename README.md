# 🚖 New York City Taxi Trip Duration Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#requirements)  
[![Status](https://img.shields.io/badge/status-completed-brightgreen.svg)](#project-status)

> Predicting taxi trip durations in New York City using geospatial feature engineering and machine learning.

---

## Table of Contents
- [Project Overview](#project-overview)  
- [Problem Statement](#problem-statement)  
- [Dataset](#dataset)  
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
- [Feature Engineering](#feature-engineering)  
- [Modeling & Evaluation](#modeling--evaluation)
- [Future Work](#future-work)   

---

## Project Overview
This repository contains a full pipeline to predict taxi trip durations for NYC yellow taxis. The pipeline includes:
- data cleaning and exploratory analysis  
- geospatial and temporal feature engineering (Haversine, Manhattan distances, cluster features, time features)  
- model training and evaluation (baseline → XGBoost/LightGBM)  
- scripts for inference and model export

**Goal:** Build a model that accurately predicts the trip duration (in seconds) given pickup/dropoff locations, timestamps, and other meta features.

---

## Problem Statement
Given features about a taxi trip (pickup time, pickup/dropoff coordinates, passenger count, vendor, etc.), predict the trip duration in seconds. The metric used for evaluation is **Root Mean Squared Log Error (RMSLE)**, which penalizes relative differences and handles skewed duration distributions.

---

## Dataset
**Source:** Kaggle - *New York City Taxi Trip Duration* (you may place the original link in your repo).  
Typical fields in the training set:

- `id` — unique trip id  
- `vendor_id` — provider id (categorical)  
- `pickup_datetime` — start timestamp  
- `dropoff_datetime` — end timestamp (train only)  
- `passenger_count` — integer  
- `pickup_longitude`, `pickup_latitude` — float  
- `dropoff_longitude`, `dropoff_latitude` — float  
- `store_and_fwd_flag` — Y/N  
- `trip_duration` — target (seconds)

**Notes & pre-processing suggestions**
- Remove trips with invalid coordinates or zero distance.  
- Cap extremely long durations (outliers) or treat them separately.  
- Convert timestamps to timezone-aware datetimes if needed.  
- Consider merging external data (weather, holidays) for improved performance.

---

## Exploratory Data Analysis (EDA)
Recommended analyses and plots included in the notebook:
- Distribution of trip durations (log-scale)  
- Pickup/dropoff location heatmaps (overlaid on NYC map)  
- Trips per hour / weekday seasonality plots  
- Relationship between distance and duration (scatter + lowess)  
- Passenger count vs duration; vendor id breakdowns  
- Distance vs speed (to detect outliers/inaccurate records)

---

## Feature Engineering
Key engineered features used in this project:

### Temporal features
- `pickup_hour`, `pickup_minute`, `pickup_weekday`, `pickup_day`, `pickup_month`  
- `is_weekend`, `is_holiday`, `is_rush_hour` (custom rule)  
- cyclical encoding for hour/day with `sin`/`cos` transforms

### Geospatial features
- **Haversine distance** between pickup and dropoff (great-circle)
- **Manhattan distance** approximation (sum of lat/lon differences after transformation)
- **Bearing** (direction) from pickup → dropoff
- `pickup_cluster`, `dropoff_cluster` (KMeans on coordinates)
- `distance_to_city_center` (distance to a reference point like Times Square)
- `neighbor_count` / local density via 2D KDE or grid bucketing

#### Example Haversine function (Python)
```python
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    # all args in decimal degrees
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # distance in km
```

## Modeling & Evaluation
### Metric
- Primary: RMSLE (Root Mean Squared Log Error)
```text
RMSLE = sqrt( (1/n) * sum( (log(pred + 1) - log(actual + 1))^2 ) )
```
### Cross-validation

- Use K-Fold (e.g., 5-fold) with care to avoid time leakage
- Optionally use TimeSeriesSplit if you want strict time-order training/validation

### Models tried

- Baseline: Mean or median predictor, Linear Regression on log-transformed target
- Tree-based: Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Ensemble: Stacking of top models
- Neural models: (optional) fully-connected or sequence-based networks for advanced experiments

### Hyperparameter tuning

- Use RandomizedSearchCV or Optuna for efficient tuning
- Try early stopping for boosting models for speed and generalization

### Interpretability

- Feature importance from tree models
- SHAP (SHapley Additive exPlanations) for insights per-prediction

## Future Work

- Integrate weather, traffic, and event data for improved accuracy.
- Build a lightweight web UI for interactive predictions and visualization.
- Try neural sequence models or graph-based approaches for spatial modelling.
- Explore more advanced spatial clustering (HDBSCAN) or graph routing distances.
