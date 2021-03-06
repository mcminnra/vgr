#!/usr/bin/env python3

import ast
from datetime import date
import logging
import pathlib

from hyperopt import fmin, hp, atpe, Trials, STATUS_OK
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import shap
from xgboost import XGBRegressor
import yaml

from db import init_games, get_games, get_reviews
from etl import get_data, process_data

# Globals
today = date.today()

# Logging
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger('vgr')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# Config
with open(str(pathlib.Path(__file__).parent.parent.absolute()) + '/config.yml', "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == '__main__':
    # ==============================================================================================
    # Get Data
    # ==============================================================================================
    logger.debug('Refreshing database')
    init_games()

    # ==============================================================================================
    # Data processing and Feature Engineering
    # ==============================================================================================
    logger.debug('Starting process_data()')
    
    # Get games and add reviews
    df_games = get_games()
    df_reviews = get_reviews()
    df_reviews = df_reviews.rename(columns={'title': 'input_name', 'rating': 'personal_rating'})[['igdb_id', 'steam_id', 'input_name', 'personal_rating']]
    df = df_games.merge(df_reviews, how='left', on=['igdb_id', 'steam_id'])

    df = process_data(df)

    # ==============================================================================================
    # Model Training
    # ==============================================================================================
    logger.debug('Starting model training')
    print('\n=== Model ===')
    rating_idx = df['personal_rating'].notnull()
    df_train = df[rating_idx]
    df_pred = df[-rating_idx]
    print(f'Training Data - Rows:{df_train.shape[0]}, Cols:{df_train.shape[1]}')

    feature_cols = [col for col in df.columns if 'feat_' in col]
    X_train = df_train[feature_cols]
    X_pred = df_pred[feature_cols]
    y_train = df_train['personal_rating']

    # Ignore specific XGBoost error
    import warnings
    warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

    ### Train Model
    def objective(params):     
        pca = PCA(n_components=int(params['n_components']), random_state=42)
        model = XGBRegressor(
            max_depth=int(params['max_depth']),
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            reg_lambda=params['reg_lambda'],
            reg_alpha=params['reg_alpha'],
            objective='reg:squarederror',
            verbosity=0,
            seed=42,
            n_jobs=-1
        )
        X_train_pca = pca.fit_transform(X_train)
        model.fit(X_train_pca, y_train)
        loss = cross_val_score(model, X_train_pca, y_train, scoring='neg_mean_squared_error', cv=5).mean()*-1

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': params
        }

    best = fmin(
        fn=objective,
        space={
            'n_components': hp.quniform('n_components', 1, min(*X_train.shape), 1),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),  # Default = 6
            'subsample': hp.uniform('subsample', 0.2, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1.0),
            'min_child_weight': hp.uniform('min_child_weight', 0, 2),
            'reg_lambda': hp.uniform('reg_lambda', 0, 2),  # Default = 1
            'reg_alpha': hp.uniform('reg_alpha', 0, 4),  # Default = 0
        },
        algo=atpe.suggest,
        max_evals=1000,
        trials=Trials()  # allows us to record info from each iteration
    )
    print(best)

    # Use best params
    pca = PCA(n_components=int(best['n_components']), random_state=42)
    model = XGBRegressor(
        max_depth=int(best['max_depth']),
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        min_child_weight=best['min_child_weight'],
        reg_lambda=best['reg_lambda'],
        reg_alpha=best['reg_alpha'],
        objective='reg:squarederror',
        verbosity=0,
        seed=42,
        n_jobs=-1
    )
    X_train_pca = pca.fit_transform(X_train)
    model.fit(X_train_pca, y_train)

    y_hat = model.predict(X_train_pca)
    scores = cross_val_score(model, X_train_pca, y_train, scoring='neg_mean_squared_error', cv=5)
    print(f'Avg. MSE: {scores.mean()*-1:0.4f} (+/- {scores.std():0.4f})')

    y_pred = model.predict(pca.transform(X_pred))

    ### Train Shap Approx. Model
    def objective_shap(params):     
        model = XGBRegressor(
            max_depth=int(params['max_depth']),
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            reg_lambda=params['reg_lambda'],
            reg_alpha=params['reg_alpha'],
            objective='reg:squarederror',
            verbosity=0,
            seed=42,
            n_jobs=-1
        )
        model.fit(X_pred, y_pred)
        loss = cross_val_score(model, X_pred, y_pred, scoring='neg_mean_squared_error', cv=5).mean()*-1

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': params
        }

    best = fmin(
        fn=objective_shap,
        space={
            'max_depth': hp.quniform('max_depth', 3, 15, 1),  # Default = 6
            'subsample': hp.uniform('subsample', 0.2, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1.0),
            'min_child_weight': hp.uniform('min_child_weight', 0, 2),
            'reg_lambda': hp.uniform('reg_lambda', 0, 2),  # Default = 1
            'reg_alpha': hp.uniform('reg_alpha', 0, 4),  # Default = 0
        },
        algo=atpe.suggest,
        max_evals=15,
        trials=Trials()  # allows us to record info from each iteration
    )
    print(best)

    # Use best params
    model_shap = XGBRegressor(
        max_depth=int(best['max_depth']),
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        min_child_weight=best['min_child_weight'],
        reg_lambda=best['reg_lambda'],
        reg_alpha=best['reg_alpha'],
        objective='reg:squarederror',
        verbosity=0,
        seed=42,
        n_jobs=-1
    )
    model_shap.fit(X_pred, y_pred)
    explainer = shap.TreeExplainer(model_shap)
    shap_values = explainer(X_pred)
    df_shap = pd.DataFrame(shap_values.values, columns=X_pred.columns)

    corrs = []
    for col in X_pred.columns:
        corr, _ = stats.spearmanr(X_pred[col], df_shap[col])
        corr = corr if not np.isnan(corr) else 0
        corrs.append((col, corr))

    c_i = [(col, corr, df_shap[col].abs().mean()) for col, corr in corrs]
    c_i = sorted(c_i, key=lambda x: x[1]*x[2], reverse=True)
    c_i = [(col, corr, shap_imp) for col, corr, shap_imp in c_i if 'text' not in col]  # Remove text embedding importance

    ### Output
    df_pred['predicted_rating'] = y_pred
    df_pred = df_pred[['input_name', 'predicted_rating']].sort_values('predicted_rating', ascending=False)

    # Games
    print('\n== Top 25 Games ==')
    for index, row in df_pred.head(25).iterrows():
        print(f'{row["input_name"]}: {row["predicted_rating"]:0.2f}')

    print('\n== Bottom 25 Games ==')
    for index, row in df_pred.tail(25).iterrows():
        print(f'{row["input_name"]}: {row["predicted_rating"]:0.2f}')

    print('\n == Random Game ==')
    rand_int = np.random.randint(low=0, high=df_pred.shape[0])
    print(f'{df_pred.iloc[rand_int]["input_name"]}: {df_pred.iloc[rand_int]["predicted_rating"]:0.2f}')

    # Shap Corr+Magnitude
    print('\n== Positive Features by Correlation+Importance ==')
    for col, corr, shap_imp in c_i[:10]:
        print(f'{col}: {corr:0.3f}//{shap_imp:0.3f}')

    print('\n== Negative Features by Correlation+Importance ==')
    for col, corr, shap_imp in c_i[-10:]:
        print(f'{col}: {corr:0.3f}//{shap_imp:0.3f}')

    print('\n== Positive Meta Features by Correlation+Importance ==')
    for col, corr, shap_imp in c_i[:10]:
        if 'meta' in col:
            print(f'{col}: {corr:0.3f}//{shap_imp:0.3f}')

    print('\n== Negative meta Features by Correlation+Importance ==')
    for col, corr, shap_imp in c_i[-10:]:
        if 'meta' in col:
            print(f'{col}: {corr:0.3f}//{shap_imp:0.3f}')
    
    # Output File
    df_pred.to_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + f'/data/scores_{today.year}_{today.month}_{today.day}.csv')
