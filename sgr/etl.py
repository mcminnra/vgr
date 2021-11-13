from ast import literal_eval
from datetime import date, timedelta
import os
import pathlib

import numpy as np
import pandas as pd
from rich import print
from rich.progress import track
from sentence_transformers import SentenceTransformer

from steamapi import get_library_appids, get_wishlist_appids, get_steam_search_appids, get_store_data

# Globals
base_path = str(pathlib.Path(__file__).parent.parent.absolute())


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

    # Get popular new releases appids
    appids_pnr = get_steam_search_appids('https://store.steampowered.com/search/?filter=popularnew&sort_by=Released_DESC&os=win')
    df_pnr = pd.DataFrame({'Steam AppID': appids_pnr, 'Rating':None})
    df_pnr = df_pnr.astype({'Steam AppID': int})
    print(f'Number of AppIDs Found in Steam "Popular New Releases": {df_pnr.shape[0]}')

    # Get top sellers
    appids_ts = get_steam_search_appids('https://store.steampowered.com/search/?filter=topsellers&os=win')
    df_ts = pd.DataFrame({'Steam AppID': appids_ts, 'Rating':None})
    df_ts = df_ts.astype({'Steam AppID': int})
    print(f'Number of AppIDs Found in Steam "Top Sellers": {df_ts.shape[0]}')

    # Join appids
    df = df.append([df_library, df_wishlist, df_pnr, df_ts])
    df = df.rename(columns={"Steam AppID": "steam_appid", "Rating": "rating"}).drop_duplicates(subset=['steam_appid'], keep='first')
    df = df.set_index('steam_appid')

    # Load cache
    cache_path = base_path + '/data/cache.csv'
    if os.path.isfile(cache_path):
        df_cache = pd.read_csv(cache_path).set_index('steam_appid')
    else:
        df_cache = pd.DataFrame(
            columns=[
                'steam_appid',
                '_date_pulled',
                'name',
                'release_date',
                'recent_count',
                'recent_percent',
                'all_count',
                'all_percent',
                'short_desc',
                'long_desc',
                'tags',
                'is_dlc',
                'is_soundtrack',
                'is_video',
            ]).set_index('steam_appid')

    # Enrich appids
    for appid in track(df.index.values, description='Getting Steam Store Data'):
        # Not in cache
        if appid not in df_cache.index:
            dict_appid = get_store_data(appid)
            s_appid = pd.Series(dict_appid).rename(dict_appid['steam_appid']).drop(labels=['steam_appid'])
            df_cache = df_cache.append(s_appid)
        # Outdated cache
        elif date.fromisoformat(df_cache.at[appid,'_date_pulled']) < (date.today() - timedelta(days=14)):
            dict_appid = get_store_data(appid)
            s_appid = pd.Series(dict_appid).rename(dict_appid['steam_appid']).drop(labels=['steam_appid'])
            df_cache = df_cache.drop([appid]).append(s_appid)

    # resolve types
    df_cache = df_cache.convert_dtypes()
    df_cache['tags'] = df_cache['tags'].apply(lambda x: literal_eval(x) if x and isinstance(x, str) else x)

    # write cache
    df_cache.sort_values(by='name').to_csv(cache_path)

    # Join with ratings    
    df = df_cache.merge(df, how='left', right_index=True, left_index=True, suffixes=(None, None))
    del df_cache

    print(f'\nTotal AppIDs in cache: {df.shape[0]}')

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
    df = df[(pd.to_datetime(df['release_date']).dt.date <= date.today()) & (df['release_date'].notnull())]
    after = df.shape[0]
    print(f'Removed {before-after} AppIDs - Not Released')

    # Summary
    print(f'Total AppIDs in filtered dataset: {df.shape[0]}')
    print(f'Total AppIDs reviewed: {df[df["rating"].notnull()].shape[0]}')

    return df

def process_data(df):
    """
    Process data and do some feature engineering

    Note: prefix "feat_" is columns used for training
    """

    ### Fill Null
    df['short_desc'] = df['short_desc'].fillna('')
    df['long_desc'] = df['short_desc'].fillna('')

    ### Explode Tags
    df['tags'] = df['tags'].apply(lambda row: [item.replace(' ', '_').replace('-', '_').replace('\'', '').replace('.', 'point') for item in row])
    df_tags = pd.get_dummies(df[['tags']].explode('tags')).sum(level=0)
    df_tags = df_tags.add_prefix('feat_')
    df = df.merge(df_tags, how='inner', right_index=True, left_index=True, suffixes=(None, None))

    ### Embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 100

    # Create embedding and explode short_desc embedding to multiple cols
    df['feat_emb_short_desc'] = df['short_desc'].apply(lambda x: model.encode(x))
    emb_len = df['feat_emb_short_desc'].values[0].shape[0]
    emb_cols = [f'feat_emb_short_desc_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['feat_emb_short_desc'].tolist(), index=df.index)
    df = df.drop(['feat_emb_short_desc'], axis=1).copy()

    # Create embedding and explode long_desc embedding to multiple cols
    df['feat_emb_long_desc'] = df['long_desc'].apply(lambda x: model.encode(x))
    emb_len = df['feat_emb_long_desc'].values[0].shape[0]
    emb_cols = [f'feat_emb_long_desc_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['feat_emb_long_desc'].tolist(), index=df.index)
    df = df.drop(['feat_emb_long_desc'], axis=1).copy()

    ### Feature transforms
    transform_cols = [col for col in df.columns if 'percent' in col or 'count' in col]

    # Log
    df_log = np.log(df[transform_cols])
    df_log = df_log.copy().fillna(0).add_suffix('_log')
    df_log[np.isinf(df_log)] = 0

    # Pow
    df_pow = np.power(df[transform_cols], 2)
    df_pow = df_pow.copy().fillna(0).add_suffix('_pow')
    df_pow[np.isinf(df_pow)] = 0
    
    df = df.merge(df_log, how='inner', right_index=True, left_index=True)
    df = df.merge(df_pow, how='inner', right_index=True, left_index=True)
    del df_log, df_pow

    ### Normalize
    normalize_cols = [col for col in df.columns if 'percent' in col or 'count' in col]
    for col in normalize_cols:
        df[f'feat_norm_{col}']=((df[f'{col}']-df[f'{col}'].mean())/df[f'{col}'].std()).fillna(0)
    df = df.astype({f'{col}': float for col in df.columns if 'feat_norm_' in col})  # Convert to float because it gets changed for some reason

    return df.copy()
