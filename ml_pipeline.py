"""Shared input contract and scikit-learn Pipeline architecture with zero data leakage."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC = ['area', 'front_width', 'n_bedrooms', 'n_bathrooms', 'n_floors']
CATEGORICAL = ['city_province', 'district', 'property_type', 'legal', 'interior']
FEATURES = NUMERIC + CATEGORICAL


def prepare_inputs(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    x = frame[FEATURES].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors='raise').astype(float)
        if np.isinf(x[col]).any() or x[col].dropna().lt(0).any():
            raise ValueError(f'Invalid numeric input: {col}')
    if x.area.isna().any() or x.area.le(0).any():
        raise ValueError('Area must be positive and present')
    for col in CATEGORICAL:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    # District names are meaningful only within their province/city.
    x['district'] = (x.city_province.fillna('Unknown').astype(str) + ' | ' +
                     x.district.fillna('Unknown').astype(str))
    return x


def make_model(estimator, scale=False):
    numeric = [('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True))]
    if scale:
        numeric.append(('scale', StandardScaler()))
    pre = ColumnTransformer([
        ('numeric', Pipeline(numeric), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL)])
    pipeline = Pipeline([('input', FunctionTransformer(prepare_inputs)),
                         ('preprocess', pre), ('model', estimator)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
