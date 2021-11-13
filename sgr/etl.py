from ast import literal_eval
from datetime import date

import numpy as np
import pandas as pd
from rich import print
from rich.progress import track
from sentence_transformers import SentenceTransformer

from steamapi import get_library_appids, get_wishlist_appids, get_store_data

def get_data(df, steam_url_name, steam_id):
    total_games = df.shape[0]

    # Process scores df
    df = df[df['Steam AppID'].notnull()]
    df = df.astype({'Steam AppID': int})
    df = df[['Steam AppID', 'Rating']]
    print(f'Number of AppIDs Found in Review File: {df.shape[0]} (out of {total_games})')

    # Get library appids
    appids_library = get_library_appids(steam_url_name)
    df_library = pd.DataFrame({'Steam AppID': appids_library, 'Rating':None})
    df_library = df_library.astype({'Steam AppID': int})
    print(f'Number of AppIDs Found in Steam Library: {df_library.shape[0]}')

    # Get wishlist appids
    appids_wishlist = get_wishlist_appids(steam_id)
    df_wishlist = pd.DataFrame({'Steam AppID': appids_wishlist, 'Rating':None})
    df_wishlist = df_wishlist.astype({'Steam AppID': int})
    print(f'Number of AppIDs Found in Steam Wishlist: {df_wishlist.shape[0]}')

    # Join appids
    df = df.merge(df_library, on='Steam AppID', how='outer', suffixes=('', '_y'))[['Steam AppID', 'Rating']]
    df = df.merge(df_wishlist, on='Steam AppID', how='outer', suffixes=('', '_y'))[['Steam AppID', 'Rating']]
    df = df.set_index('Steam AppID')

    # Enrich appids
    for appid in track(df.index.values, description='Enriching with Steam Store Data'):
        data_appid = get_store_data(appid)
        keys = data_appid.keys()

        for key in keys:
            if key not in df.columns:
                df[key] = None
            df.at[appid, key] = data_appid[key]

    # Check for valid data points
    # No Names
    before = df.shape[0]
    df = df[df['name'].notnull()]
    after = df.shape[0]
    print(f'Removed {before-after} AppIDs - Missing Store Data')

    # DLC Check
    before = df.shape[0]
    df = df[df['is_dlc'] == False]
    after = df.shape[0]
    print(f'Removed {before-after} AppIDs - DLC')

    # Soundtrack Check
    before = df.shape[0]
    df = df[df['is_soundtrack'] == False]
    after = df.shape[0]
    print(f'Removed {before-after} AppIDs - Soundtrack')

    # Video Check
    before = df.shape[0]
    df = df[df['is_video'] == False]
    after = df.shape[0]
    print(f'Removed {before-after} AppIDs - Video')

    # Not Released Check
    before = df.shape[0]
    today = date.today()
    df = df[(df['release_date'] <= today) & (df['release_date'].notnull())]
    after = df.shape[0]
    print(f'Removed {before-after} AppIDs - Not Released')

    # Summary
    print(f'Total AppIDs in dataset: {df.shape[0]}')
    print(f'Num of AppIDs reviewed: {df[df["Rating"].notnull()].shape[0]}')

    return df

def process_data(df):
    """
    Process data and do some feature engineering

    Note: prefix "feat_" is columns used for training
    """
    df = df.set_index('Steam AppID')

    # Fill Null
    df['short_desc'] = df['short_desc'].fillna('')
    df['long_desc'] = df['short_desc'].fillna('')

    # Normalize ratings cols
    df['feat_norm_recent_percent']=(df['recent_percent']-df['recent_percent'].mean())/df['recent_percent'].std()
    df['feat_norm_recent_count']=(df['recent_count']-df['recent_count'].mean())/df['recent_count'].std()
    df['feat_norm_all_percent']=(df['all_percent']-df['all_percent'].mean())/df['all_percent'].std()
    df['feat_norm_all_count']=(df['all_count']-df['all_count'].mean())/df['all_count'].std()

    # Explode Tags
    df.tags = df.tags.apply(literal_eval)
    df_tags = pd.get_dummies(df[['tags']].explode('tags')).sum(level=0)
    df_tags = df_tags.add_prefix('feat_')
    df = df.merge(df_tags, how='inner', right_index=True, left_index=True, suffixes={None, None})

    ### Embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 300
    # Process short_desc
    # Get short_desc embeddings
    df['feat_emb_short_desc'] = None
    for idx, sentence in zip(df.index, df['short_desc'].values):
        sentence_emb = model.encode(sentence)
        df.at[idx, 'feat_emb_short_desc'] = sentence_emb

    # Explode short_desc_emb to multiple cols
    emb_len = df['feat_emb_short_desc'].values[0].shape[0]
    emb_cols = [f'feat_emb_short_desc_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['feat_emb_short_desc'].tolist(), index=df.index)
    df = df.drop(['feat_emb_short_desc'], axis=1)

    # Process long_desc
    # Get long_desc embeddings
    df['feat_emb_long_desc'] = None
    for idx, sentence in zip(df.index, df['long_desc'].values):
        sentence_emb = model.encode(sentence)
        df.at[idx, 'feat_emb_long_desc'] = sentence_emb

    # Explode short_desc_emb to multiple cols
    emb_len = df['feat_emb_long_desc'].values[0].shape[0]
    emb_cols = [f'feat_emb_long_desc_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['feat_emb_long_desc'].tolist(), index=df.index)
    df = df.drop(['feat_emb_long_desc'], axis=1)

    ### Feature transforms
    feature_cols = [col for col in df.columns if 'feat_' in col]

    # Log
    df_log = np.log(df[feature_cols]).fillna(0).add_suffix('_log')
    df_log[df_log == -np.inf] = 0

    # Pow
    df_pow = np.power(df[feature_cols], 2).fillna(0).add_suffix('_pow')
    df_pow[df_pow == np.inf] = 0
    
    df = df.merge(df_log, how='inner', right_index=True, left_index=True)
    df = df.merge(df_pow, how='inner', right_index=True, left_index=True)
    del df_log, df_pow

    return df
