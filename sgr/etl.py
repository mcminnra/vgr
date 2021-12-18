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
    df.to_csv(data_path, index=False)
    print(df)

    return df


def process_data(df):
    """
    Process data and do some feature engineering

    Note: prefix "feat_" is columns used for training
    """
    # Filter to found IGDB Games
    df = df[df['igdb_id'].notnull()].set_index('igdb_id').replace({np.nan: None})

    ### Clean Data
    # Drop unneeded cols
    df = df.drop([
            'igdb_name',
            'igdb_status',
            'igdb_first_release_date',
            'igdb_keywords',
            'steam_name'
        ], axis=1)

    # Description
    df['igdb_summary'] = df['igdb_summary'].fillna('')
    df['igdb_storyline'] = df['igdb_storyline'].fillna('')
    df['steam_short_desc'] = df['steam_short_desc'].fillna('')

    # Ratings
    df['igdb_critics_rating'] = df['igdb_critics_rating'].fillna(df['igdb_critics_rating'].mean()).round(0).astype(int)
    df['igdb_critics_rating_count'] = df['igdb_critics_rating_count'].fillna(0).round(0).astype(int)
    df['igdb_user_rating'] = df['igdb_user_rating'].fillna(df['igdb_user_rating'].mean()).round(0).astype(int)
    df['igdb_user_rating_count'] = df['igdb_user_rating_count'].fillna(0).round(0).astype(int)
    df['steam_recent_percent'] = df['steam_recent_percent'].fillna(df['steam_recent_percent'].mean()).round(0).astype(int)
    df['steam_recent_count'] = df['steam_recent_count'].fillna(df['steam_recent_count'].mean()).round(0).astype(int)
    df['steam_all_percent'] = df['steam_all_percent'].fillna(df['steam_all_percent'].mean()).round(0).astype(int)
    df['steam_all_count'] = df['steam_all_count'].fillna(df['steam_all_count'].mean()).round(0).astype(int)

    # Tags
    tag_replace = lambda item: item.replace(' ', '_').replace('-', '_').replace('\'', '').replace('.', 'point').replace('&', 'and')
    df['igdb_genres'] = df['igdb_genres'].apply(lambda row: [tag_replace(item) for item in row] if row else None)
    df['igdb_themes'] = df['igdb_themes'].apply(lambda row: [tag_replace(item) for item in row] if row else None)
    df['steam_tags'] = df['steam_tags'].apply(lambda row: [tag_replace(item) for item in row] if row else None)

    ### Feature Engineering
    # Convert to categorical to numeric labels
    df['igdb_category'] = df['igdb_category'].astype('category').cat.codes

    # Create combined desc column
    df['desc'] = df['igdb_summary'] + df['igdb_storyline'] + df['steam_short_desc']
    df = df.drop(['igdb_summary', 'igdb_storyline', 'steam_short_desc'], axis=1)

    # explode tags
    def explode_log(df, explode_col):
        df_tags = df[[explode_col]]
        UNIQUE_TAGS = sorted(list(set([tag for row in df_tags[explode_col].tolist() if row for tag in row])), reverse=False)
        df_tags[[f'{explode_col}_{tag}' for tag in UNIQUE_TAGS]] = None

        # Map tag postion importance to column
        for row_idx, row in df_tags.iterrows():
            if row[explode_col]:
                tags_len = len(row[explode_col])
                for tag_idx, tag in enumerate(row[explode_col]):
                    if tags_len == 1:
                        df_tags.at[row_idx, f'{explode_col}_{tag}'] = 1
                    else:
                        df_tags.at[row_idx, f'{explode_col}_{tag}'] = math.log(tags_len-tag_idx, tags_len)  # Logarithmic importance
        df_tags = df_tags.drop([explode_col], axis=1).fillna(0)

        return df_tags

    def explode_binary(df, explode_col):
        df_tags = df[[explode_col]]
        UNIQUE_TAGS = sorted(list(set([tag for row in df_tags[explode_col].tolist() if row for tag in row])), reverse=False)
        df_tags[[f'{explode_col}_{tag}' for tag in UNIQUE_TAGS]] = None

        # Map tag postion importance to column
        for row_idx, row in df_tags.iterrows():
            if row[explode_col]:
                for tag in row[explode_col]:
                    df_tags.at[row_idx, f'{explode_col}_{tag}'] = 1
        df_tags = df_tags.drop([explode_col], axis=1).fillna(0)

        return df_tags

    df = df.merge(explode_binary(df, 'igdb_genres'), how='inner', right_index=True, left_index=True, suffixes=(None, None)).drop(['igdb_genres'], axis=1)
    df = df.merge(explode_binary(df, 'igdb_themes'), how='inner', right_index=True, left_index=True, suffixes=(None, None)).drop(['igdb_themes'], axis=1)
    df = df.merge(explode_log(df, 'steam_tags'), how='inner', right_index=True, left_index=True, suffixes=(None, None)).drop(['steam_tags'], axis=1)
    
    # Embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 500

    df['emb_desc'] = df['desc'].progress_apply(lambda x: np.mean([model.encode(sentence) for sentence in x.split('. ')], axis=0))
    emb_len = df['emb_desc'].values[0].shape[0]
    emb_cols = [f'emb_desc_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['emb_desc'].tolist(), index=df.index)
    df = df.drop(['desc', 'emb_desc'], axis=1)

    # # Feature Transforms
    # numerical_cols = [
    #     'igdb_critics_rating',
    #     'igdb_critics_rating_count',
    #     'igdb_user_rating',
    #     'igdb_user_rating_count',
    #     'steam_recent_percent',
    #     'steam_recent_count',
    #     'steam_all_percent',
    #     'steam_all_count'
    # ]
    # normalize_cols = []
    # for col in numerical_cols:
    #     df[f'{col}_log'] = np.log(df[f'{col}'])
    #     df[f'{col}_pow'] = np.power(df[f'{col}'], 2)
    #     df[f'{col}_recip'] = 1 / df[f'{col}']
    #     df[f'{col}_sqrt'] = np.sqrt(df[f'{col}'])
    #     normalize_cols += [col, f'{col}_log', f'{col}_pow', f'{col}_recip', f'{col}_sqrt']
    # df = df.replace([np.inf, -np.inf], 0, inplace=True)

    # ### Normalize
    # normalize_cols = [col for col in df.columns if 'percent' in col or 'count' in col]
    # for col in normalize_cols:
    #     df[f'feat_norm_{col}']=((df[f'{col}']-df[f'{col}'].mean())/df[f'{col}'].std()).fillna(0)
    # df = df.astype({f'{col}': float for col in df.columns if 'feat_norm_' in col})  # Convert to float because it gets changed for some reason

    # Add "feat_" prefixes
    ignore_cols = ['input_name', 'steam_id', 'personal_rating']
    rename_cols = {col: f'feat_{col}' for col in df.columns if col not in ignore_cols}
    df.rename(columns=rename_cols, inplace=True)

    ### Write out processed for debugging
    processed_path = base_path + '/data/processed.csv'
    df.to_csv(processed_path, index=False)

    print(df)

    return df
