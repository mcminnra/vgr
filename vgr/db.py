import logging
import pathlib
import sqlite3
import yaml

import pandas as pd
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

igdb_client = IGDBClient(config['igdb_client_id'], config['igdb_client_secret'])
steam_client = SteamClient(config['steam_url_name'], config['steam_user_id'], config['steam_web_api_key'])


def init_games():
    with sqlite3.connect('/mnt/g/My Drive/video_games/video_games.db') as con:
        # Reviews db
        df_review_ids = pd.read_sql_query("SELECT * from reviews", con)[['igdb_id', 'steam_id']]
        # Non-Steam Backlog
        df_nsb_ids = pd.read_sql_query("SELECT * from non_steam_backlog", con)[['igdb_id', 'steam_id']]
        
        # Steam Library
        steam_library_ids = steam_client.get_steam_library_ids()
        df_steam_library_ids = pd.DataFrame({'steam_id': steam_library_ids})
        df_steam_library_ids['igdb_id'] = None
        
        # Steam Wishlist
        steam_wishlist_ids = steam_client.get_steam_wishlist_ids()
        df_steam_wishlist_ids = pd.DataFrame({'steam_id': steam_wishlist_ids})
        df_steam_wishlist_ids['igdb_id'] = None
        df = pd.concat([df_review_ids, df_nsb_ids, df_steam_library_ids, df_steam_wishlist_ids], axis=0)

        # Map IGDB and Steam
        with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
            for i, row in tqdm(df.iterrows(), desc='Mapping IGDB and Steam IDs'):
                if row['igdb_id'] and not row['steam_id']:  # Only IGDB id set. Pull Steam ID from website link.
                    df.at[i, 'steam_id'] = igdb_client.get_steam_id_by_igdb_id(row['igdb_id'])
                elif not row['igdb_id'] and row['steam_id']:  # Only Steam ID set. Pull IGDB ID from games that use the Steam ID website link.
                    df.at[i, 'igdb_id'] = igdb_client.get_igdb_id_by_steam_id(row['steam_id'])

        # Drop duplicates+na
        df = df.drop_duplicates(subset=['igdb_id', 'steam_id']).dropna()

        # TODO Pull in previous games and do a diff

        # Write the new DataFrame to a new SQLite table
        df.to_sql("games", con, if_exists="replace", index=False)
        

if __name__ == '__main__':
    init_games()






