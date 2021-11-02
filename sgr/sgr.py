#!/usr/bin/env python3

import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd
from rich import print
from rich.console import Console
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBRegressor

from etl import get_data, preprocess_data

# CLI
parser = argparse.ArgumentParser(description='Steam Games Recommender')
parser.add_argument(
    'reviews_filepath',
    type=str,
    help='Reviews input filepath that has a "Steam AppID" and "Rating" columns.'
)

if __name__ == '__main__':
    # get print console
    console = Console()

    # Get args
    args = parser.parse_args()

    ### ETL
    # Get reviewed games
    print(f'Reviews Input Path: {args.reviews_filepath}')
    df = pd.read_excel(args.reviews_filepath)

    # Create data by getting library and wishlist games and enriching input
    df = get_data(df)
    df.to_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/raw.csv')  # cache
    del df

    ### Data Pre-processing
    df = pd.read_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/raw.csv')
    df = preprocess_data(df)
    df.to_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/processed.csv')  # cache
    del df

    ### Model Training
    df = pd.read_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/processed.csv').set_index('Steam AppID')
    
    normalized_cols = [col for col in df.columns if '_norm' in col]
    tag_cols = [col for col in df.columns if 'tags_' in col]
    embedding_cols = [col for col in df.columns if '_emb' in col]

    df_train = df[df['Rating'].notnull()]
    df_pred = df[df['Rating'].isnull()]

    X_train = df_train[normalized_cols+tag_cols+embedding_cols]
    X_pred = df_pred[normalized_cols+tag_cols+embedding_cols]
    y_train = df_train['Rating']

    # Fit Model
    model = XGBRegressor(
        max_depth=4,  # 32
        n_estimators=354,  # 250
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
        'AppID': X_pred.index.values,
        'Pred Score': y_pred
    }).sort_values('Pred Score', ascending=False).set_index('AppID')
    df_pred = df_pred.join(df[['name']], how='left')
    df_pred = df_pred[['name', 'Pred Score']]

    # Top 10
    print('\n== Top 10 ==')
    for index, row in df_pred.head(10).iterrows():
        print(f'{row["name"]}: {row["Pred Score"]:0.2f}')

    # Bottom 10
    print('\n== Bottom 10 ==')
    for index, row in df_pred.tail(10).iterrows():
        print(f'{row["name"]}: {row["Pred Score"]:0.2f}')

    print('\n== Feature Importances ==')
    feat_imp = [(col, imp) for col, imp in zip(X_train.columns, model.feature_importances_)]

    tag_imp = np.mean([imp for col, imp in feat_imp if 'tags_' in col])
    short_desc_imp = np.mean([imp for col, imp in feat_imp if 'short_desc_' in col])
    long_desc_imp = np.mean([imp for col, imp in feat_imp if 'long_desc_' in col])

    print(f'Tag Avg. Importance: {tag_imp:0.6f}')
    print(f'Short Desc. Avg. Importance: {short_desc_imp:0.6f}')
    print(f'Long Desc Avg. Importance: {long_desc_imp:0.6f}')
    for col, imp in feat_imp:
        if '_norm' in col:
            print(f'{col} Importance: {imp:0.6f}')

    print('\nTop 10 Tags')
    tags = [(name, imp) for name, imp in feat_imp if 'tags_' in name]
    for name, imp in sorted(tags, key=lambda x: x[1], reverse=True)[:10]:
        print(name, imp)

    pickle.dump(model, open(str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/model.pkl', 'wb'))
    
