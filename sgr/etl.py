from ast import literal_eval
from datetime import date, timedelta
import math
import os
import pathlib
import pickle

from igdb.wrapper import IGDBWrapper
import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from game import Game
from steamapi import get_steam_library, get_steam_wishlist, get_library_appids, get_wishlist_appids, get_steam_search_appids, get_store_data

# Globals
base_path = str(pathlib.Path(__file__).parent.parent.absolute())
tqdm.pandas()


def get_data(config):
    # Get IGDB
    # igdb_client_id = "hlax4aqmbgg9ke7pu6ijaajemr92uk"
    # igdb_client_secret = "xpmve0632748x2vl4iu7srl80ecl5z"
    # r = requests.post(f"https://id.twitch.tv/oauth2/token?client_id={igdb_client_id}&client_secret={igdb_client_secret}&grant_type=client_credentials")
    # print(r.text)

    igdb_client = IGDBWrapper("hlax4aqmbgg9ke7pu6ijaajemr92uk", "2jq81qzucxhfmgwbkkcdnvzj1nxu5w")

    # Load cache if exists
    cache_path = base_path + '/data/cache.pkl'
    if os.path.isfile(cache_path):
        print(f'Loaded cache at => {cache_path}')
        with open(cache_path, 'rb') as file:
            games = pickle.load(file)
    else:
        games = []

    # Get Reviews File
    df = pd.read_excel(config['reviews_filepath']).replace({np.nan: None})
    for _, row in tqdm(df.iterrows(), desc='Getting Review File'):
        game_igdb_ids = [game.igdb_id for game in games]
        if row['IGDB ID'] not in game_igdb_ids:
            game = Game(
                igdb_client,
                name=row['Name'],
                igdb_id=row['IGDB ID'],
                steam_id=row['Steam ID'],
                personal_rating=row['Rating']
            )
            games.append(game)

    # Get Steam Library
    for item in tqdm(get_steam_library(config['steam_url_name']), desc='Getting Steam Library'):
        game_steam_ids = [game.steam_id for game in games]
        if item['steam_id'] not in game_steam_ids:
            game = Game(
                igdb_client,
                name=item['name'],
                steam_id=item['steam_id'],
            )
            games.append(game)

    # Get Steam Wishlist
    for item in tqdm(get_steam_wishlist(config['steam_user_id']), desc='Getting Steam Wishlist'):
        game_steam_ids = [game.steam_id for game in games]
        if item['steam_id'] not in game_steam_ids:
            game = Game(
                igdb_client,
                name=item['name'],
                steam_id=item['steam_id']
            )
            games.append(game)

    # Dump Cache
    with open(cache_path, 'wb') as file:
        pickle.dump(games, file)

    # Convert to df
    df = pd.concat([game.to_series() for game in games], axis=1, ignore_index=True).T
    data_path = base_path + '/data/data.csv'
    df.to_csv(data_path)
    print(df)

    return df


def process_data(df):
    """
    Process data and do some feature engineering

    Note: prefix "feat_" is columns used for training
    """
    import sys; sys.exit()
    pass


def old_process_data(df):
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
