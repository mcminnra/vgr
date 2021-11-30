#!/usr/bin/env python3

from datetime import date
import pathlib
import pickle

from hyperopt import fmin, hp, tpe, atpe, Trials, STATUS_OK
import numpy as np
import pandas as pd
from rich import print
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import yaml

from etl import get_data, process_data

# Globals
today = date.today()
console = Console()

# Load config
with open(str(pathlib.Path(__file__).parent.parent.absolute()) + '/config.yml', "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == '__main__':
    # ==============================================================================================
    # Get Data
    # ==============================================================================================
    print('=== Getting Data ===')
    # Get reviewed games
    print(f'Reviews Input Path: {config["reviews_filepath"]}')
    df = pd.read_excel(config['reviews_filepath'])

    # Create data by getting library and wishlist games and enriching input
    df = get_data(df, config['steam_url_name'], config['steam_id'])

    # ==============================================================================================
    # Data processing and Feature Engineering
    # ==============================================================================================
    print('\n=== Processing Data ===')
    df = process_data(df)

    # ==============================================================================================
    # Model Training
    # ==============================================================================================
    print('\n=== Model ===')
    rating_idx = df['rating'].notnull()
    df_train = df[rating_idx]
    df_pred = df[-rating_idx]
    print(f'Training Data - Rows:{df_train.shape[0]}, Cols:{df_train.shape[1]}')

    feature_cols = [col for col in df.columns if 'feat_' in col]
    X_train = df_train[feature_cols]
    X_pred = df_pred[feature_cols]
    y_train = df_train['rating']

    # Ignore specific XGBoost error
    import warnings
    warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

    # Hyperparam tuning
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
        mse = cross_val_score(model, X_train_pca, y_train, scoring='neg_mean_squared_error', cv=5).mean()*-1

        return {
            'loss': mse,
            'status': STATUS_OK,
            'params': params
        }

    search_space = {
        'n_components': hp.quniform('n_components', 1, min(*X_train.shape), 1),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),  # Default = 6
        'subsample': hp.uniform('subsample', 0.2, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1.0),
        'min_child_weight': hp.uniform('min_child_weight', 0, 2),
        'reg_lambda': hp.uniform('reg_lambda', 0, 2),  # Default = 1
        'reg_alpha': hp.uniform('reg_alpha', 0, 4),  # Default = 0
    }

    trials = Trials()  # allows us to record info from each iteration
    best = fmin(
        fn=objective,
        space=search_space,
        algo=atpe.suggest,
        max_evals=2000,
        trials=trials
    )
    print(best)

    # Train Model
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
    scores = cross_val_score(model, X_train_pca, y_train, scoring='neg_mean_squared_error', cv=5)
    print(f'Avg. MSE: {scores.mean()*-1:0.4f} (+/- {scores.std()*-1:0.4f})')

    # Predict
    y_pred = model.predict(pca.transform(X_pred))

    df_pred = pd.DataFrame({
        'steam_appid': X_pred.index.values,
        'pred_score': y_pred
    }).set_index('steam_appid')
    df_pred = df_pred.join(df[['name']], how='left')
    df_pred = df_pred[['name', 'pred_score']].sort_values('pred_score', ascending=False)

    # Top
    print('\n== Top 25 ==')
    for index, row in df_pred.head(25).iterrows():
        print(f'{row["name"]}: {row["pred_score"]:0.2f}')

    # Bottom
    print('\n== Bottom 25 ==')
    for index, row in df_pred.tail(25).iterrows():
        print(f'{row["name"]}: {row["pred_score"]:0.2f}')

    # Random
    print('\n == Random Game ==')
    rand_int = np.random.randint(low=0, high=df_pred.shape[0])
    print(f'{df_pred.iloc[rand_int]["name"]}: {df_pred.iloc[rand_int]["pred_score"]:0.2f}')
    
    # Output
    df_pred.to_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + f'/data/scores_{today.year}_{today.month}_{today.day}.csv')
