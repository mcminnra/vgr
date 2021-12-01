from ast import literal_eval
from datetime import date, timedelta
import math
import os
import pathlib

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from steamapi import get_library_appids, get_wishlist_appids, get_steam_search_appids, get_store_data

# Globals
base_path = str(pathlib.Path(__file__).parent.parent.absolute())
tqdm.pandas()


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
                'reviews_text',
                'tags',
                'is_dlc',
                'is_soundtrack',
                'is_video',
            ]).set_index('steam_appid')

    # Enrich appids
    for appid in tqdm(df.index.values, desc='Getting Steam Store Data'):
        retries = 0
        while retries < 5:
            try:
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
            except Exception:
                print(f'Error getting appid {appid}. Retry {retries+1}')
                retries+=1
                continue
            break

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
    df['reviews_text'] = df['reviews_text'].fillna('')

    ### Explode Tags and get position importance
    # Filter tags
    df['tags'] = df['tags'].apply(lambda row: [item.replace(' ', '_').replace('-', '_').replace('\'', '').replace('.', 'point').replace('&', 'and') for item in row])

    # Get sert of unique tags and create cols
    df_tags = df[['tags']]
    UNIQUE_TAGS = sorted(list(set([tag for row in df_tags['tags'].tolist() for tag in row])), reverse=False)
    df_tags[[f'feat_tags_{tag}' for tag in UNIQUE_TAGS]] = None

    # Map tag postion importance to column
    for row_idx, row in df_tags.iterrows():
        tags_len = len(row['tags'])
        for tag_idx, tag in enumerate(row['tags']):
            df_tags.at[row_idx, f'feat_tags_{tag}'] = math.log(tags_len-tag_idx, tags_len)  # Logarithmic importance
    df_tags = df_tags.drop(['tags'], axis=1).fillna(0)

    # Join back to df
    df = df.merge(df_tags, how='inner', right_index=True, left_index=True, suffixes=(None, None))

    ### Embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 100

    # Create embedding and explode desc embedding to multiple cols
    print(f'Encoding Description Embeddings...')
    df['desc'] = df['short_desc'] + ' ' + df['long_desc']
    df['feat_emb_desc'] = df['desc'].progress_apply(lambda x: np.mean([model.encode(sentence) for sentence in x.split('. ')], axis=0))
    emb_len = df['feat_emb_desc'].values[0].shape[0]
    emb_cols = [f'feat_emb_desc_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['feat_emb_desc'].tolist(), index=df.index)
    df = df.drop(['feat_emb_desc'], axis=1).copy()

    # Create embedding and explode reviews_test embedding to multiple cols
    print(f'Encoding Review Embeddings...')
    df['feat_emb_reviews_text'] = df['reviews_text'].progress_apply(lambda x: np.mean([model.encode(sentence) for sentence in x.split('. ')], axis=0))
    emb_len = df['feat_emb_reviews_text'].values[0].shape[0]
    emb_cols = [f'feat_emb_reviews_text_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['feat_emb_reviews_text'].tolist(), index=df.index)
    df = df.drop(['feat_emb_reviews_text'], axis=1).copy()

    ### Feature Engineering
    # Numerical Interactions
    fe_cols = [col for col in df.columns if 'percent' in col or 'count' in col]
    fe_pairs = [(col1, col2) for col1 in fe_cols for col2 in fe_cols if col1 != col2]

    for col1, col2 in tqdm(fe_pairs, desc='Numeric Interactions'):
        df[f'{col1}x{col2}'] = df[col1] * df[col2]  # Multiplicative Interaction
        df[f'{col1}+{col2}'] = df[col1] + df[col2]  # Additive Interaction

    ## Feature Transforms
    fe_cols = [col for col in df.columns if 'percent' in col or 'count' in col]
    # Log
    df_tmp = np.log(df[fe_cols])
    df_tmp = df_tmp.copy().fillna(0).add_suffix('_log')
    df_tmp[np.isinf(df_tmp)] = 0
    df = df.merge(df_tmp, how='inner', right_index=True, left_index=True)

    # Pow
    df_tmp = np.power(df[fe_cols], 2)
    df_tmp = df_tmp.copy().fillna(0).add_suffix('_pow')
    df_tmp[np.isinf(df_tmp)] = 0
    df = df.merge(df_tmp, how='inner', right_index=True, left_index=True)

    # Reciprical
    df_tmp = 1 / df[fe_cols]
    df_tmp = df_tmp.copy().fillna(0).add_suffix('_recip')
    df_tmp[np.isinf(df_tmp)] = 0
    df = df.merge(df_tmp, how='inner', right_index=True, left_index=True)

    # Sqrt
    df_tmp = np.sqrt(df[fe_cols])
    df_tmp = df_tmp.copy().fillna(0).add_suffix('_sqrt')
    df_tmp[np.isinf(df_tmp)] = 0
    df = df.merge(df_tmp, how='inner', right_index=True, left_index=True)

    del df_tmp

    ### Normalize
    normalize_cols = [col for col in df.columns if 'percent' in col or 'count' in col]
    for col in normalize_cols:
        df[f'feat_norm_{col}']=((df[f'{col}']-df[f'{col}'].mean())/df[f'{col}'].std()).fillna(0)
    df = df.astype({f'{col}': float for col in df.columns if 'feat_norm_' in col})  # Convert to float because it gets changed for some reason

    ### Write out processed for debugging
    processed_path = base_path + '/data/processed.csv'
    df.sort_values(by='name').to_csv(processed_path)

    return df
