import json
import logging
import math
import os
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from sentence_transformers import SentenceTransformer

from game import Game
from igdb_client import IGDBClient
from steam_client import SteamClient


# Logging
logger = logging.getLogger('vgr.etl')


# Globals
base_path = str(pathlib.Path(__file__).parent.parent.absolute())
tqdm.pandas()


def get_data(config):
    igdb_client = IGDBClient(config['igdb_client_id'], config['igdb_client_secret'])
    steam_client = SteamClient(config['steam_url_name'], config['steam_user_id'])

    # Load cache if exists
    cache_path = base_path + '/data/cache.pkl'
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as file:
            games = pickle.load(file)
        logger.debug(f'Loaded cache from {cache_path}')
    else:
        games = []
        logger.debug(f'No cache found.')

    # =========================================================================
    # Reviews File
    # =========================================================================
    # Get Reviews File
    df = pd.read_excel(config['reviews_filepath']).replace({np.nan: None})
    logger.info(f'{df.shape[0]} games found in review file')
    with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
        for _, row in tqdm(df.iterrows(), desc='Getting Review File'):
            # FIXME: Quick hack to convert to int
            row['IGDB ID'] = int(row['IGDB ID']) if row['IGDB ID'] else None
            row['Steam ID'] = int(row['Steam ID']) if row['Steam ID'] else None

            # Re-pull reviews data if rating different or not in games
            if row['IGDB ID'] in games:
                if row['Rating'] != games[games.index(row['IGDB ID'])].personal_rating:
                    del games[games.index(row['IGDB ID'])]
                    game = Game(
                        igdb_client,
                        steam_client,
                        name=row['Name'],
                        igdb_id=row['IGDB ID'],
                        steam_id=row['Steam ID'],
                        personal_rating=row['Rating']
                    )
                    games.append(game)
                    # FIXME: Probably shouldn't we writing after every game, but fuck it. 
                    with open(cache_path, 'wb') as file:
                        pickle.dump(games, file)
                    logger.debug(f'Reviews => Pulled {game}')
            else:
                game = Game(
                    igdb_client,
                    steam_client,
                    name=row['Name'],
                    igdb_id=row['IGDB ID'],
                    steam_id=row['Steam ID'],
                    personal_rating=row['Rating']
                )
                games.append(game)
                # FIXME: Probably shouldn't we writing after every game, but fuck it. 
                with open(cache_path, 'wb') as file:
                    pickle.dump(games, file)
                logger.debug(f'Reviews => Pulled {game}')

    # =========================================================================
    # Steam
    # =========================================================================
    # Get Steam Library
    steam_items = steam_client.get_steam_library()
    logger.info(f'{len(steam_items)} games found in Steam library')
    with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
        for item in tqdm(steam_items, desc='Getting Steam Library'):
            game_steam_ids = [game.steam_id for game in games]
            if item['steam_id'] not in game_steam_ids:
                game = Game(
                    igdb_client,
                    steam_client,
                    name=item['name'],
                    steam_id=item['steam_id'],
                )
                games.append(game)
                # FIXME: Probably shouldn't we writing after every game, but fuck it. 
                with open(cache_path, 'wb') as file:
                    pickle.dump(games, file)
                logger.debug(f'Steam Library => Pulled {game}')

    # Get Steam Wishlist
    steam_items = steam_client.get_steam_wishlist()
    logger.info(f'{len(steam_items)} games found in Steam wishlist')
    with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
        for item in tqdm(steam_items, desc='Getting Steam Wishlist'):
            game_steam_ids = [game.steam_id for game in games]
            if item['steam_id'] not in game_steam_ids:
                game = Game(
                    igdb_client,
                    steam_client,
                    name=item['name'],
                    steam_id=item['steam_id']
                )
                games.append(game)
                # FIXME: Probably shouldn't be writing after every game, but fuck it. 
                with open(cache_path, 'wb') as file:
                    pickle.dump(games, file)
                logger.debug(f'Steam Wishlist => Pulled {game}')

    # Popular New Releases
    steam_items = steam_client.get_steam_search_page('https://store.steampowered.com/search/?filter=popularnew&sort_by=Released_DESC&os=win')
    logger.info(f'{len(steam_items)} games found in Steam Popular New Releases')
    with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
        for item in tqdm(steam_items, desc='Getting Steam Popular New Releases'):
            game_steam_ids = [game.steam_id for game in games]
            if item['steam_id'] not in game_steam_ids:
                game = Game(
                    igdb_client,
                    steam_client,
                    name=item['name'],
                    steam_id=item['steam_id']
                )
                games.append(game)
                # FIXME: Probably shouldn't be writing after every game, but fuck it. 
                with open(cache_path, 'wb') as file:
                    pickle.dump(games, file)
                logger.debug(f'Steam Popular New Releases => Pulled {game}')

    # Top Sellers
    steam_items = steam_client.get_steam_search_page('https://store.steampowered.com/search/?filter=topsellers&os=win')
    logger.info(f'{len(steam_items)} games found in Steam Top Sellers')
    with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
        for item in tqdm(steam_items, desc='Getting Steam Top Sellers'):
            game_steam_ids = [game.steam_id for game in games]
            if item['steam_id'] not in game_steam_ids:
                game = Game(
                    igdb_client,
                    steam_client,
                    name=item['name'],
                    steam_id=item['steam_id']
                )
                games.append(game)
                # FIXME: Probably shouldn't be writing after every game, but fuck it. 
                with open(cache_path, 'wb') as file:
                    pickle.dump(games, file)
                logger.debug(f'Steam Top Sellers => Pulled {game}')

    # =========================================================================
    # IGDB
    # =========================================================================
    ### IGDB Top 100
    # Pulling for all platforms
    byte_array = igdb_client._igdb_wrapper.api_request(
        'platforms',
        f'fields id, name; limit 500;'
    )
    platforms_response = json.loads(byte_array)

    for i, platform in enumerate(platforms_response):
        byte_array = igdb_client._igdb_wrapper.api_request(
            'games',
            f'fields id, name; where platforms = {platform["id"]} & total_rating != null & total_rating_count >= 5; sort total_rating desc; limit 100;'
        )
        time.sleep(0.25)
        games_response = json.loads(byte_array)
        if not games_response:
            logger.debug(f'No rated games found for {platform["name"]}.')
            continue
        with logging_redirect_tqdm(loggers=[logging.getLogger('vgr')]):
            for game in tqdm(games_response, desc=f'[{i+1}/{len(platforms_response)}] Getting IGDB {platform["name"]} Top 100'):
                game_igdb_ids = [game.igdb_id for game in games]
                if game['id'] not in game_igdb_ids:
                    game = Game(
                        igdb_client,
                        steam_client,
                        name=game['name'],
                        igdb_id=game['id'],
                    )
                    games.append(game)
                    # FIXME: Probably shouldn't we writing after every game, but fuck it. 
                    with open(cache_path, 'wb') as file:
                        pickle.dump(games, file)
                    logger.debug(f'IGDB Top 100 ({platform["name"]}) => Pulled {game}')

    # Convert to df
    df = pd.concat([game.to_series() for game in games], axis=1, ignore_index=True).T
    data_path = base_path + '/data/data.csv'
    df.to_csv(data_path, index=False)
    logger.debug(f'Wrote data to {data_path}')
    print(df)

    return df


def process_data(df):
    """
    Process data and do some feature engineering

    Note: prefix "feat_" is columns used for training
    """
    # Drop unneeded cols
    df = df.drop([
            'igdb_name',
            'igdb_status',
            'igdb_first_release_date',
            'steam_name'
        ], axis=1)

    # =========================================================================
    # Data Clean
    # =========================================================================
    ### Filters
    df = df[df['igdb_category'] != 'dlc_addon']

    ### Fill na
    df['igdb_user_rating'] = df['igdb_user_rating'].fillna(0).round(0).astype(int)
    df['igdb_user_rating_count'] = df['igdb_user_rating_count'].fillna(0).round(0).astype(int)

    df['igdb_summary'] = df['igdb_summary'].fillna('')
    df['igdb_storyline'] = df['igdb_storyline'].fillna('')
    df['steam_short_desc'] = df['steam_short_desc'].fillna('')

    df['igdb_genres'] = df['igdb_genres'].apply(lambda d: d if isinstance(d, list) else [])
    df['igdb_keywords'] = df['igdb_keywords'].apply(lambda d: d if isinstance(d, list) else [])
    df['igdb_themes'] = df['igdb_themes'].apply(lambda d: d if isinstance(d, list) else [])
    df['steam_tags'] = df['steam_tags'].apply(lambda d: d if isinstance(d, list) else [])

    # =========================================================================
    # Coalesce Cols
    # =========================================================================
    ### Rating
    # TODO: Should normalize
    df['rating'] = df['steam_all_percent'].fillna(df['igdb_user_rating']).round(0).astype(int)

    ### Popularity
    df['igdb_rating_count_perc'] = df['igdb_user_rating_count'].rank(pct=True)
    df['steam_all_count_perc'] = df['steam_all_count'].rank(pct=True)
    df['popularity'] = df['steam_all_count_perc'].fillna(df['igdb_rating_count_perc']).round(2).astype(float)

    ### Tags
    df['tags'] = df['igdb_genres'] + df['igdb_keywords'] + df['igdb_themes'] + df['steam_tags']
    df['tags'] = df['tags'].apply(lambda X: list(set([x.lower().replace('/', ' ').replace('(', '').replace(')', '').replace('\'', '').replace(',', '').replace('<', '').replace('>', '').replace(':', '').replace('[', '').replace(']', '') for x in X])))

    # Handcraft some "meta" tags
    UNIQUE_TAGS = sorted(list(set([tag for row in df['tags'].tolist() if row for tag in row])), reverse=False)
    meta_tags = {
        # Dev Size
        'meta_indie': [tag for tag in UNIQUE_TAGS if 'indie' in tag],
        # Genre / Gameplay
        'meta_arcade': [tag for tag in UNIQUE_TAGS if 'arcade' in tag],
        'meta_action': [tag for tag in UNIQUE_TAGS if 'action' in tag],
        'meta_adventure': [tag for tag in UNIQUE_TAGS if 'adventure' in tag],
        'meta_building': [tag for tag in UNIQUE_TAGS if 'building' in tag],
        'meta_casual': [tag for tag in UNIQUE_TAGS if 'casual' in tag],
        'meta_coop': [tag for tag in UNIQUE_TAGS if 'co-op' in tag or 'coop' in tag or 'co op' in tag],
        'meta_crafting': [tag for tag in UNIQUE_TAGS if 'crafting' in tag],
        'meta_difficult': [tag for tag in UNIQUE_TAGS if 'difficult' in tag or 'hard' in tag],
        'meta_exploration': [tag for tag in UNIQUE_TAGS if 'exploration' in tag],
        'meta_fighting': [tag for tag in UNIQUE_TAGS if 'fighting' in tag],
        'meta_hack_and_slash': [tag for tag in UNIQUE_TAGS if 'hack and slash' in tag or 'hack-and-slash' in tag],
        'meta_management': [tag for tag in UNIQUE_TAGS if 'management' in tag],
        'meta_music': [tag for tag in UNIQUE_TAGS if 'music' in tag],
        'meta_open_world': [tag for tag in UNIQUE_TAGS if 'open world' in tag or 'open-world' in tag],
        'meta_racing': [tag for tag in UNIQUE_TAGS if 'racing' in tag],
        'meta_roguelike': [tag for tag in UNIQUE_TAGS if 'roguelike' in tag or 'rogue-like' in tag or 'roguelite' in tag or 'rogue-lite' in tag],
        'meta_rpg': [tag for tag in UNIQUE_TAGS if 'rpg' in tag or 'role playing' in tag or 'role-playing' in tag],
        'meta_rts': [tag for tag in UNIQUE_TAGS if 'rts' in tag],
        'meta_party': [tag for tag in UNIQUE_TAGS if 'party' in tag or 'minigames' in tag],
        'meta_platformer': [tag for tag in UNIQUE_TAGS if 'platformer' in tag],
        'meta_puzzle': [tag for tag in UNIQUE_TAGS if 'puzzle' in tag],
        'meta_sandbox': [tag for tag in UNIQUE_TAGS if 'sandbox' in tag],
        'meta_shooter': [tag for tag in UNIQUE_TAGS if 'shooter' in tag or 'fps' in tag],
        'meta_simulation': [tag for tag in UNIQUE_TAGS if 'simulation' in tag or 'simulator' in tag],
        'meta_sports': [tag for tag in UNIQUE_TAGS if 'sport' in tag],
        'meta_stealth': [tag for tag in UNIQUE_TAGS if 'stealth' in tag],
        'meta_strategy': [tag for tag in UNIQUE_TAGS if 'strategy' in tag],
        'meta_survival': [tag for tag in UNIQUE_TAGS if 'survival' in tag],
        'meta_tactical': [tag for tag in UNIQUE_TAGS if 'tactical' in tag],
        'meta_turn_based': [tag for tag in UNIQUE_TAGS if 'turn-based' in tag or 'turn based' in tag],
        # Style / Setting
        'meta_first_person': [tag for tag in UNIQUE_TAGS if 'first-person' in tag or 'first person' in tag],
        'meta_third_person': [tag for tag in UNIQUE_TAGS if 'third-person' in tag or 'third person' in tag],
        'meta_2d': [tag for tag in UNIQUE_TAGS if '2d' in tag or '2.5d' in tag],
        'meta_3d': [tag for tag in UNIQUE_TAGS if '3d' in tag],
        'meta_aliens': [tag for tag in UNIQUE_TAGS if 'alien' in tag],
        'meta_anime': [tag for tag in UNIQUE_TAGS if 'anime' in tag],
        'meta_atmospheric': [tag for tag in UNIQUE_TAGS if 'atmospheric' in tag],
        'meta_comedy': [tag for tag in UNIQUE_TAGS if 'comedy' in tag or 'funny' in tag],
        'meta_fantasy': [tag for tag in UNIQUE_TAGS if 'fantasy' in tag],
        'meta_gore': [tag for tag in UNIQUE_TAGS if 'blood' in tag or 'gore' in tag or 'violent' in tag],
        'meta_historical': [tag for tag in UNIQUE_TAGS if 'historical' in tag or 'history' in tag],
        'meta_horror': [tag for tag in UNIQUE_TAGS if 'horror' in tag],
        'meta_magic': [tag for tag in UNIQUE_TAGS if 'magic' in tag],
        'meta_medieval': [tag for tag in UNIQUE_TAGS if 'medieval' in tag],
        'meta_narrative': [tag for tag in UNIQUE_TAGS if 'narrative' in tag or 'story rich' in tag or 'multiple endings'],
        'meta_pixel': [tag for tag in UNIQUE_TAGS if 'pixel' in tag],
        'meta_retro': [tag for tag in UNIQUE_TAGS if 'retro' in tag or 'classic' in tag],
        'meta_post_apocalyptic': [tag for tag in UNIQUE_TAGS if 'post-apocalyptic' in tag or 'post apocalyptic' in tag],
        'meta_robots': [tag for tag in UNIQUE_TAGS if 'robots' in tag],
        'meta_scifi': [tag for tag in UNIQUE_TAGS if 'sci-fi' in tag or 'science fiction' in tag],
        'meta_soundtrack': [tag for tag in UNIQUE_TAGS if 'soundtrack' in tag],
        'meta_space': [tag for tag in UNIQUE_TAGS if 'space' in tag],
        'meta_sword': [tag for tag in UNIQUE_TAGS if 'sword' in tag],
    }
    for idx, row in df.iterrows():
        for meta_tag in meta_tags.keys():
            if list(set(row['tags']) & set(meta_tags[meta_tag])):
                df.at[idx, 'tags'] += [meta_tag]

    # Count Tags
    # tag_count = {}
    # for _, row in df.iterrows():
    #     for tag in row['tags']:
    #         if tag in tag_count:
    #             tag_count[tag] += 1
    #         else:
    #             tag_count[tag] = 1

    # for k, v in sorted(tag_count.items(), key=lambda item: item[0]):
    #     print(k, v)
    
    ### Text
    df['text'] = df['igdb_summary'] + df['igdb_storyline'] + df['steam_short_desc']

    df = df[['input_name', 'igdb_id', 'steam_id', 'personal_rating', 'rating', 'popularity', 'tags', 'text']]

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    ### Explode tags
    def explode_log(df, explode_col):
        df_tags = df[[explode_col]]
        UNIQUE_TAGS = sorted(list(set([tag for row in df_tags[explode_col].tolist() if row for tag in row])), reverse=False)
        df_tags[[f'{explode_col}_{tag}' for tag in UNIQUE_TAGS]] = 0

        # Map tag postion importance to column
        for row_idx, row in df_tags.iterrows():
            if row[explode_col]:
                tags_len = len(row[explode_col])
                for tag_idx, tag in enumerate(row[explode_col]):
                    if tags_len == 1:
                        df_tags.at[row_idx, f'{explode_col}_{tag}'] = 1
                    else:
                        df_tags.at[row_idx, f'{explode_col}_{tag}'] = math.log(tags_len-tag_idx, tags_len)  # Logarithmic importance
        df_tags = df_tags.drop([explode_col], axis=1)
        return df_tags

    def explode_binary(df, explode_col):
        df_tags = df[[explode_col]]
        UNIQUE_TAGS = sorted(list(set([tag for row in df_tags[explode_col].tolist() if row for tag in row])), reverse=False)
        df_tags[[f'{explode_col}_{tag}' for tag in UNIQUE_TAGS]] = 0

        # Map tag postion importance to column
        for row_idx, row in df_tags.iterrows():
            if row[explode_col]:
                for tag in row[explode_col]:
                    df_tags.at[row_idx, f'{explode_col}_{tag}'] = 1
        df_tags = df_tags.drop([explode_col], axis=1)
        return df_tags

    df = df.merge(explode_binary(df, 'tags'), how='inner', right_index=True, left_index=True, suffixes=(None, None)).drop(['tags'], axis=1).drop_duplicates()

    # Drop all tags that aren't meta
    keep_cols = [col for col in df.columns if 'tags' not in col or 'tags_meta_' in col]
    df = df[keep_cols]

    ### Embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 500

    df['emb_text'] = df['text'].progress_apply(lambda x: np.mean([model.encode(sentence) for sentence in x.split('. ')], axis=0))
    emb_len = df['emb_text'].values[0].shape[0]
    emb_cols = [f'emb_text_{i}' for i in range(0, emb_len)]
    df[emb_cols] = pd.DataFrame(df['emb_text'].tolist(), index=df.index)
    df = df.drop(['text', 'emb_text'], axis=1)

    # Add "feat_" prefixes
    ignore_cols = ['input_name', 'igdb_id', 'steam_id', 'personal_rating']
    rename_cols = {col: f'feat_{col}' for col in df.columns if col not in ignore_cols}
    df.rename(columns=rename_cols, inplace=True)

    ### Post-filter
    # NOTE: We mainly do this to remove games that haven't been released yet. These will have rating of 0 and will be overwhelmingly negatively rated
    df = df[df['feat_rating'] > 0]

    ### Write out processed for debugging
    processed_path = base_path + '/data/processed.csv'
    df.to_csv(processed_path, index=False)

    print(df)

    return df
