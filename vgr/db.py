from enum import Enum
import logging
import pathlib
import sqlite3
import yaml

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


from igdb_client import IGDBClient
from steam_client import SteamClient

# Logging
logger = logging.getLogger('vgr.db')
tqdm.pandas()

# Config
with open(str(pathlib.Path(__file__).parent.parent.absolute()) + '/config.yml', "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Data Clients
igdb_client = IGDBClient(config['igdb_client_id'], config['igdb_client_secret'])
steam_client = SteamClient(config['steam_url_name'], config['steam_user_id'], config['steam_web_api_key'])

# Database engine
engine = create_engine("sqlite:////mnt/g/My Drive/video_games/video_games.db", echo=True, future=True)


class IGDBGameCategory(Enum):
    main_game=0
    dlc_addon=1
    expansion=2
    bundle=3
    standalone_expansion=4
    mod=5
    episode=6
    season=7
    remake=8
    remaster=9
    expanded_game=10
    port=11
    fork=12


class IGDBGameStatus(Enum):
    released=0
    alpha=2
    beta=3
    early_access=4
    offline=5
    cancelled=6
    rumored=7
    delisted=8

def get_igdb_data(igdb_id):
    igdb_metadata = igdb_client.get_game(igdb_id)

    # Rename id
    igdb_metadata['igdb_id'] = igdb_metadata.pop('id')

    # Process game metadata
    igdb_metadata['igdb_name'] = igdb_metadata['name'] if 'name' in igdb_metadata else None
    igdb_metadata['igdb_category'] = IGDBGameCategory(igdb_metadata['category']).name if 'category' in igdb_metadata else None
    igdb_metadata['igdb_status'] = IGDBGameStatus(igdb_metadata['status']).name if 'status' in igdb_metadata else None
    igdb_metadata['igdb_first_release_date'] = igdb_metadata['first_release_date'] if 'first_release_date' in igdb_metadata else None
    igdb_metadata['igdb_critics_rating'] = igdb_metadata['aggregated_rating'] if 'aggregated_rating' in igdb_metadata else None
    igdb_metadata['igdb_critics_rating_count'] = igdb_metadata['aggregated_rating_count'] if 'aggregated_rating_count' in igdb_metadata else None
    igdb_metadata['igdb_user_rating'] = igdb_metadata['rating'] if 'rating' in igdb_metadata else None
    igdb_metadata['igdb_user_rating_count'] = igdb_metadata['rating_count'] if 'rating_count' in igdb_metadata else None
    igdb_metadata['igdb_summary'] = igdb_metadata['summary'].strip().replace('\n', '') if 'summary' in igdb_metadata else None
    igdb_metadata['igdb_storyline'] = igdb_metadata['storyline'].strip().replace('\n', '') if 'storyline' in igdb_metadata else None

    # Genres
    igdb_metadata['igdb_genres'] = igdb_client.get_genres_by_genre_ids(igdb_metadata["genres"]) if 'genres' in igdb_metadata else None
        
    # Keywords
    igdb_metadata['igdb_keywords'] = igdb_client.get_keywords_by_keyword_ids(igdb_metadata["keywords"]) if 'keywords' in igdb_metadata else None

    # Themes
    igdb_metadata['igdb_themes'] = igdb_client.get_themes_by_theme_ids(igdb_metadata["themes"]) if 'themes' in igdb_metadata else None

    # Get subset of keys
    keys = [
        'igdb_id',
        'igdb_name',
        'igdb_category',
        'igdb_status',
        'igdb_first_release_date',
        'igdb_critics_rating',
        'igdb_critics_rating_count',
        'igdb_user_rating',
        'igdb_user_rating_count',
        'igdb_summary',
        'igdb_storyline',
        'igdb_genres',
        'igdb_keywords',
        'igdb_themes'
    ]
    igdb_metadata = {k: igdb_metadata[k] for k in keys}

    return igdb_metadata


def get_steam_data(steam_id):
    steam_metadata = {'steam_id': steam_id}
    steam_store_tree = steam_client.get_steam_store_html(steam_id)

    try:
        steam_metadata['steam_name'] = steam_client.get_name_from_html(steam_store_tree)
        steam_metadata['steam_recent_percent'], steam_metadata['steam_recent_count'], steam_metadata['steam_all_percent'], steam_metadata['steam_all_count'] = steam_client.get_reviews_from_html(steam_store_tree)
        steam_metadata['steam_short_desc'] = steam_client.get_short_desc_from_html(steam_store_tree)
        steam_metadata['steam_tags'] = steam_client.get_tags_from_html(steam_store_tree)
    except Exception as e:
        logger.error(f'Failed pulling Steam metadata for {steam_id} - Does https://store.steampowered.com/app/{steam_id} exist?')

    return steam_metadata


def get_games():
    with engine.connect() as con:
        return pd.read_sql_query(text("SELECT * from games"), con)


def init_games():
    with engine.connect() as con:
        # Reviews db
        df_review_ids = pd.read_sql_query(text("SELECT * from reviews"), con)[['igdb_id', 'steam_id']]

        # Non-Steam Backlog
        df_nsb_ids = pd.read_sql_query(text("SELECT * from non_steam_backlog"), con)[['igdb_id', 'steam_id']]
        
        # Steam Library & Wishlist
        steam_library_ids = steam_client.get_steam_library_ids()
        df_steam_library_ids = pd.DataFrame({'steam_id': steam_library_ids}).astype(int)
        steam_wishlist_ids = steam_client.get_steam_wishlist_ids()
        df_steam_wishlist_ids = pd.DataFrame({'steam_id': steam_wishlist_ids}).astype(int)

        # Merge
        df = df_review_ids.merge(df_nsb_ids, how='outer', on=['igdb_id', 'steam_id'])
        df = df.merge(df_steam_library_ids, how='outer', on ='steam_id')
        df = df.merge(df_steam_wishlist_ids, how='outer', on ='steam_id')
        df = df.fillna(0).astype(int)

        # Only get data for games not in "games" table
        df_games_igdb_ids = pd.read_sql_query(text("SELECT igdb_id from games"), con)['igdb_id'].values
        df = df[~df['igdb_id'].isin(df_games_igdb_ids)]
        df_games_steam_ids = pd.read_sql_query(text("SELECT steam_id from games"), con)['steam_id'].values
        df = df[~df['steam_id'].isin(df_games_steam_ids)]

        # Map IGDB and Steam (0 == "no id")
        with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
            for i, row in tqdm(df.iterrows(), desc='Mapping IGDB and Steam IDs'):
                if row['igdb_id'] != 0 and row['steam_id'] == 0:  # Only IGDB id set. Pull Steam ID from website link.
                    df.at[i, 'steam_id'] = igdb_client.get_steam_id_by_igdb_id(int(row['igdb_id']))
                elif row['igdb_id'] == 0 and row['steam_id'] != 0:  # Only Steam ID set. Pull IGDB ID from games that use the Steam ID website link.
                    df.at[i, 'igdb_id'] = igdb_client.get_igdb_id_by_steam_id(int(row['steam_id']))
        df = df.drop_duplicates(subset=['igdb_id', 'steam_id'])

        # Grab IGDB metadata
        igdb_ids = df['igdb_id'].drop_duplicates().dropna().values
        igdb_metadata = []
        with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
            for id in tqdm(igdb_ids, desc='Pulling IGDB data'):
                igdb_metadata.append(get_igdb_data(int(id)))

        if igdb_metadata:
            df_igdb = pd.DataFrame(igdb_metadata)
            df_igdb[['igdb_genres', 'igdb_keywords', 'igdb_themes']] = df_igdb[['igdb_genres', 'igdb_keywords', 'igdb_themes']].astype(str)
            df = df.merge(df_igdb, how='left', on='igdb_id')

        # Grab Steam metadata
        steam_ids = df['steam_id'].drop_duplicates().dropna().values
        steam_metadata = []
        with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
            for id in tqdm(steam_ids, desc='Pulling Steam data'):
                steam_metadata.append(get_steam_data(int(id)))

        if steam_metadata:
            df_steam = pd.DataFrame(steam_metadata)
            df_steam[['steam_tags']] = df_steam[['steam_tags']].astype(str)
            df = df.merge(df_steam, how='left', on='steam_id')
        
        # Insert rows into games
        if not df.empty:
            con.execute(
                text(
                    """
                    INSERT OR IGNORE INTO games (
                                            igdb_id,
                                            steam_id,
                                            igdb_name,
                                            igdb_category,
                                            igdb_status,
                                            igdb_first_release_date,
                                            igdb_critics_rating,
                                            igdb_critics_rating_count,
                                            igdb_user_rating,
                                            igdb_user_rating_count,
                                            igdb_summary,
                                            igdb_storyline,
                                            igdb_genres,
                                            igdb_keywords,
                                            igdb_themes,
                                            steam_name,
                                            steam_recent_percent,
                                            steam_recent_count,
                                            steam_all_percent,
                                            steam_all_count,
                                            steam_short_desc,
                                            steam_tags
                                        ) VALUES (
                                            :igdb_id,
                                            :steam_id,
                                            :igdb_name,
                                            :igdb_category,
                                            :igdb_status,
                                            :igdb_first_release_date,
                                            :igdb_critics_rating,
                                            :igdb_critics_rating_count,
                                            :igdb_user_rating,
                                            :igdb_user_rating_count,
                                            :igdb_summary,
                                            :igdb_storyline,
                                            :igdb_genres,
                                            :igdb_keywords,
                                            :igdb_themes,
                                            :steam_name,
                                            :steam_recent_percent,
                                            :steam_recent_count,
                                            :steam_all_percent,
                                            :steam_all_count,
                                            :steam_short_desc,
                                            :steam_tags
                                        )
                    """
                ),
                df.to_dict(orient='records')
            )
            con.commit()
        

if __name__ == '__main__':
    init_games()






