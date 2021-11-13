#!/usr/bin/env python3

from datetime import date
import pathlib
import pickle

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

    feature_cols = [col for col in df.columns if 'feat_' in col]
    X_train = df_train[feature_cols]
    X_pred = df_pred[feature_cols]
    y_train = df_train['rating']

    # Fit Model
    model = XGBRegressor(
        max_depth=32,  # 32
        n_estimators=250,  # 250
        objective='reg:squarederror',
        random_state=42,
        verbosity=0,
        n_jobs=-1)
    model.fit(X_train, y_train)

    # Get Cross Val Score
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
