
from enum import Enum
import json
import requests
import time

import pandas as pd

from steamapi import get_steam_store_html, get_name_from_html, get_reviews_from_html, get_short_desc_from_html, get_tags_from_html


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


class Game:
    """
    Class to resolve various client ids to an IGDM entry and collect metadata from multiple sources.
    """
    def __init__(self, igdb_client, name, igdb_id=None, steam_id=None, personal_rating=None):        
        # Clients
        self._igdb_client = igdb_client

        # IDs
        self.input_name = name
        self.igdb_id = igdb_id
        self.steam_id = steam_id
        self._resolve_ids()

        # Metadata
        self.last_updated = int(time.time())
        self.personal_rating = personal_rating
        
        # IGDB Metadata
        self.igdb_metadata = {}
        if self.igdb_id:
            self._resolve_igdb_metadata()

        # Steam Metadata
        self.steam_metadata = {}
        if self.steam_id:
            self._resolve_steam_metadata()

    def __str__(self):
        return f'{self.input_name} (IGDB: {self.igdb_id}, Steam: {self.steam_id})'

    def __repr__(self):
        return f'{self.input_name} (IGDB: {self.igdb_id}, Steam: {self.steam_id})'

    def __eq__(self, other):
        if self.igdb_id and other.igdb_id and self.igdb_id == other.igdb_id:
            return True
        else:
            return False

    def _resolve_ids(self):
        if not self.igdb_id and not self.steam_id:  # No ids set. Return best search from IGDB
            self.igdb_id = self._igdb_client.get_igdb_id_by_name(self.input_name)
            self.steam_id = self._igdb_client.get_steam_id_by_igdb_id(self.igdb_id)
        elif self.igdb_id and not self.steam_id:  # Only IGDB id set. Pull Steam ID from website link.
            self.steam_id = self._igdb_client.get_steam_id_by_igdb_id(self.igdb_id)
        elif not self.igdb_id and self.steam_id:  # Only Steam ID set. Pull IGDB ID from games that use the Steam ID website link.
            self.igdb_id = self._igdb_client.get_igdb_id_by_steam_id(self.steam_id)
        else:  # Keys already set.
            pass  # May log something here in the future

    def _resolve_igdb_metadata(self):
        igdb_metadata = self._igdb_client.get_game(self.igdb_id)

        # Process game metadata
        self.igdb_metadata['igdb_name'] = igdb_metadata['name'] if 'name' in igdb_metadata else None
        self.igdb_metadata['igdb_category'] = IGDBGameCategory(igdb_metadata['category']).name if 'category' in igdb_metadata else None
        self.igdb_metadata['igdb_status'] = IGDBGameStatus(igdb_metadata['status']).name if 'status' in igdb_metadata else None
        self.igdb_metadata['igdb_first_release_date'] = igdb_metadata['first_release_date'] if 'first_release_date' in igdb_metadata else None
        self.igdb_metadata['igdb_critics_rating'] = igdb_metadata['aggregated_rating'] if 'aggregated_rating' in igdb_metadata else None
        self.igdb_metadata['igdb_critics_rating_count'] = igdb_metadata['aggregated_rating_count'] if 'aggregated_rating_count' in igdb_metadata else None
        self.igdb_metadata['igdb_user_rating'] = igdb_metadata['rating'] if 'rating' in igdb_metadata else None
        self.igdb_metadata['igdb_user_rating_count'] = igdb_metadata['rating_count'] if 'rating_count' in igdb_metadata else None
        self.igdb_metadata['igdb_summary'] = igdb_metadata['summary'].strip().replace('\n', '') if 'summary' in igdb_metadata else None
        self.igdb_metadata['igdb_storyline'] = igdb_metadata['storyline'].strip().replace('\n', '') if 'storyline' in igdb_metadata else None

        # Genres
        self.igdb_metadata['igdb_genres'] = self._igdb_client.get_genres_by_genre_ids(igdb_metadata["genres"]) if 'genres' in igdb_metadata else None
            
        # Keywords
        self.igdb_metadata['igdb_keywords'] = self._igdb_client.get_keywords_by_keyword_ids(igdb_metadata["keywords"]) if 'keywords' in igdb_metadata else None

        # Themes
        self.igdb_metadata['igdb_themes'] = self._igdb_client.get_themes_by_theme_ids(igdb_metadata["themes"]) if 'themes' in igdb_metadata else None
       
    def _resolve_steam_metadata(self):
        steam_store_tree = get_steam_store_html(self.steam_id)

        try:
            self.steam_metadata['steam_name'] = get_name_from_html(steam_store_tree)
            self.steam_metadata['steam_recent_percent'], self.steam_metadata['steam_recent_count'], self.steam_metadata['steam_all_percent'], self.steam_metadata['steam_all_count'] = get_reviews_from_html(steam_store_tree)
            self.steam_metadata['steam_short_desc'] = get_short_desc_from_html(steam_store_tree)
            self.steam_metadata['steam_tags'] = get_tags_from_html(steam_store_tree)
        except Exception as e:
            print(f'Failed pulling Steam metadata for {self.input_name} - Does https://store.steampowered.com/app/{self.steam_id} exist?')

        time.sleep(.6)

    def to_series(self):
        series_dict = {}

        series_dict['input_name'] = self.input_name
        series_dict['igdb_id'] = self.igdb_id
        series_dict['steam_id'] = self.steam_id
        series_dict['personal_rating'] = self.personal_rating
        
        for key, val in self.igdb_metadata.items():
            series_dict[key] = val

        for key, val in self.steam_metadata.items():
            series_dict[key] = val

        return pd.Series(series_dict)
