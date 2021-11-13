#!/usr/bin/env python3

from datetime import date
import pathlib
import pickle

from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
import numpy as np
import pandas as pd
from rich import print
from rich.console import Console
from sklearn.model_selection import GridSearchCV, cross_val_score
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
        MAX_DEPTH = params['max_depth']
        N_ESTIMATORS = params['n_estimators']
        LAMBDA = params['reg_lambda']
        ALPHA = params['reg_alpha']
        
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
        model.fit(X_train, y_train)
        mse = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()*-1

        return {
            'loss': mse,
            'status': STATUS_OK,
            'params': params
        }

    search_space = {
        'max_depth': hp.randint('max_depth', 1, 100),  # Default = 6
        'n_estimators': hp.randint('n_estimators', 1, 500),  # Default = 100
        'reg_lambda': hp.uniform('reg_lambda', 0, 2),  # Default = 1
        'reg_alpha': hp.uniform('reg_alpha', 0, 4),  # Default = 0
    }

    trials = Trials()  # allows us to record info from each iteration
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    # Train Model
    model = XGBRegressor(
        **best,
        objective='reg:squarederror',
        verbosity=0,
        seed=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    print(f'Avg. MSE: {scores.mean():0.4f} (+/- {scores.std():0.4f})')

    # Predict
    y_pred = model.predict(X_pred)

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

    print('\n== Feature Importances ==')
    feat_imp = [(col, imp) for col, imp in zip(X_train.columns, model.feature_importances_)]

    tag_imp = np.mean([imp for col, imp in feat_imp if 'tags' in col])
    short_desc_imp = np.mean([imp for col, imp in feat_imp if 'short_desc' in col])
    long_desc_imp = np.mean([imp for col, imp in feat_imp if 'long_desc' in col])
    recent_percent_imp = np.mean([imp for col, imp in feat_imp if 'recent_percent' in col])
    recent_count_imp = np.mean([imp for col, imp in feat_imp if 'recent_count' in col])
    all_percent_imp = np.mean([imp for col, imp in feat_imp if 'all_percent' in col])
    all_count_imp = np.mean([imp for col, imp in feat_imp if 'all_count' in col])

    summary_imp = [
        ('Tag Avg. Importance', tag_imp),
        ('Short Desc. Avg. Importance', short_desc_imp),
        ('Long Desc Avg. Importance', long_desc_imp),
        ('Recent Count Avg. Importance', recent_count_imp),
        ('Recent Percent Avg. Importance', recent_percent_imp),
        ('All Count Avg. Importance', all_count_imp),
        ('All Percent Avg. Importance', all_percent_imp)
    ]
    summary_imp = sorted(summary_imp, key=lambda x: x[1], reverse=True)
    for desc, imp in summary_imp:
        print(f'{desc}: {imp:0.6f}')

    print('\n== Top 10 Tags ==')
    tags = [(name, imp) for name, imp in feat_imp if 'tags' in name]
    for name, imp in sorted(tags, key=lambda x: x[1], reverse=True)[:10]:
        print(name, imp)

    # Output
    df_pred.to_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + f'/data/scores_{today.year}_{today.month}_{today.day}.csv')
    pickle.dump(model, open(str(pathlib.Path(__file__).parent.parent.absolute()) + f'/data/model_{today.year}_{today.month}_{today.day}.pkl', 'wb'))
