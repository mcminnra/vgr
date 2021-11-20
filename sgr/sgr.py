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
    print('\n=== Getting Data ===')
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
        N_COMPONENTS = params['n_components']
        MAX_DEPTH = params['max_depth']
        N_ESTIMATORS = params['n_estimators']
        LAMBDA = params['reg_lambda']
        ALPHA = params['reg_alpha']
        
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        model = XGBRegressor(
            max_depth=MAX_DEPTH,
            n_estimators=N_ESTIMATORS,
            reg_lambda=LAMBDA,
            reg_alpha=ALPHA,
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
        'n_components': hp.randint('n_components', 1, min(*X_train.shape)),
        'max_depth': hp.randint('max_depth', 1, 100),  # Default = 6
        'n_estimators': hp.randint('n_estimators', 1, 500),  # Default = 100
        'reg_lambda': hp.uniform('reg_lambda', 0, 2),  # Default = 1
        'reg_alpha': hp.uniform('reg_alpha', 0, 4),  # Default = 0
    }

    trials = Trials()  # allows us to record info from each iteration
    best = fmin(
        fn=objective,
        space=search_space,
        algo=atpe.suggest,
        max_evals=1000,
        trials=trials
    )
    print(best)

    # Train Model
    pca = PCA(n_components=best['n_components'], random_state=42)
    model = XGBRegressor(
        max_depth=best['max_depth'],
        n_estimators=best['n_estimators'],
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

    # Top 10
    print('\n== Top 10 ==')
    for index, row in df_pred.head(10).iterrows():
        print(f'{row["name"]}: {row["pred_score"]:0.2f}')

    # Bottom 10
    print('\n== Bottom 10 ==')
    for index, row in df_pred.tail(10).iterrows():
        print(f'{row["name"]}: {row["pred_score"]:0.2f}')

    # Output
    df_pred.to_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + f'/data/scores_{today.year}_{today.month}_{today.day}.csv')
