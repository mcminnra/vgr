from functools import wraps
import json
import pathlib
import time

from igdb.wrapper import IGDBWrapper
import requests
from tenacity import retry, stop_after_attempt, wait_fixed


# Globals
BASE_PATH = pathlib.Path(__file__).parent.parent.absolute()


class IGDBClient():
    """Client class to provide specific api functions to IGDB Wrapper"""

    def __init__(self, client_id: str, client_secret: str):
        # Init wrapper
        access_token = self._get_access_token(client_id, client_secret)
        self._igdb_wrapper = IGDBWrapper(client_id, access_token)

    def _get_access_token(self, client_id: str, client_secret: str) -> str:
        token_file = BASE_PATH / "data/token.json"
        current_unix_time = int(time.time())

        # If token file exists, check to see if still active and return
        if token_file.exists():
            with open(token_file, 'r') as f:
                access_json = json.load(f)
            if access_json['expires'] > current_unix_time:
                return access_json['access_token'] 

        # Get IGDB access token
        r = requests.post(f"https://id.twitch.tv/oauth2/token?client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials")
        access_json = json.loads(r.text)        
        access_json['expires'] = current_unix_time + access_json['expires_in'] - 1

        # Write access token
        with open(token_file, 'w') as f:
            json.dump(access_json, f)

        return access_json['access_token'] 

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_igdb_id_by_name(self, name: str) -> int:
        """
        Get IGDB ID by searching a name
        """
        search_input_name = name.encode('latin-1', 'ignore').decode('latin-1')

        byte_array = self._igdb_wrapper.api_request(
            'games',
            f'search "{search_input_name}"; fields *; where category != (5, 6, 7);'  # No mods, episodes, or seasons
        )
        games_response = json.loads(byte_array)
        igdb_id = int(games_response[0]['id'])

        time.sleep(.25) # sleep to not go over request limit

        return igdb_id

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_igdb_id_by_steam_id(self, steam_id: int) -> int:
        """
        Get IGDB ID using Steam ID
        """
        igdb_id = None

        search_string = f'/{steam_id}'
        byte_array = self._igdb_wrapper.api_request(
            'websites',
            f'fields *; where url = *"{search_string}"* & category = 13;'
        )
        websites_response = json.loads(byte_array)
        
        if len(websites_response) > 1:
            # More than 1 game has the same steam id website. Take the one with the most reviews or first released
            search_igdb_ids = str(tuple([r['game'] for r in websites_response]))
            byte_array = self._igdb_wrapper.api_request(
                'games',
                f'fields *; where id = {search_igdb_ids} & category != (5, 6, 7);'  # No mods, episodes, or seasons
            )
            games_response = json.loads(byte_array)
            for game in games_response:
                if 'total_rating_count' not in game:
                    game['total_rating_count'] = 0
                if 'first_release_date' not in game:
                    game['first_release_date'] = int(time.time())
                game['first_release_date'] *= -1

            igdb_id = int(sorted(games_response, key=lambda x: (x['total_rating_count'], x['first_release_date']), reverse=True)[0]['id'])
        elif len(websites_response) == 1:
            igdb_id = int(websites_response[0]['game'])

        time.sleep(.5) # sleep to not go over request limit

        return igdb_id

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_steam_id_by_igdb_id(self, igdb_id: int) -> int:
        """
        Get Steam ID using IGDB ID
        """
        steam_id = None

        byte_array = self._igdb_wrapper.api_request(
            'websites',
            f'fields *; where game = {igdb_id} & category = 13;'
        )
        websites_response = json.loads(byte_array)

        if websites_response:
            steam_url = websites_response[0]['url']
            steam_id = int([part for part in steam_url.split('/') if part.isnumeric()][0])

        time.sleep(.25) # sleep to not go over request limit
        
        return steam_id

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_game(self, igdb_id: int) -> dict:
        """
        Get game metadata from IGDB ID
        """
        byte_array = self._igdb_wrapper.api_request(
            'games',
            f'fields *; where id = {igdb_id};'
        )
        games_response = json.loads(byte_array)
        igdb_metadata = games_response[0]

        time.sleep(.25) # sleep to not go over request limit

        return igdb_metadata

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_genres_by_genre_ids(self, genre_ids: list[int]) -> list[str]:
        """
        Get genre names using IGDB Genre IDs
        """
        genre_id_str = str(tuple(genre_ids)) if len(genre_ids) > 1 else genre_ids[0]
        byte_array = self._igdb_wrapper.api_request(
            'genres',
            f'fields *; where id = {genre_id_str};'
        )
        genres_response = json.loads(byte_array)
        genres = [r['name'] for r in genres_response]

        time.sleep(.25) # sleep to not go over request limit

        return genres

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_keywords_by_keyword_ids(self, keyword_ids: list[int]) -> list[str]:
        """
        Get keyword names using IGDB keyword IDs
        """
        keyword_id_str = str(tuple(keyword_ids)) if len(keyword_ids) > 1 else keyword_ids[0]
        byte_array = self._igdb_wrapper.api_request(
            'keywords',
            f'fields *; where id = {keyword_id_str};'
        )
        keywords_response = json.loads(byte_array)
        keywords = [r['name'] for r in keywords_response]

        time.sleep(.25) # sleep to not go over request limit

        return keywords

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_themes_by_theme_ids(self, theme_ids: list[int]) -> list[str]: 
        """
        Get theme names using IGDB theme IDs
        """
        theme_id_str = str(tuple(theme_ids)) if len(theme_ids) > 1 else theme_ids[0]
        byte_array = self._igdb_wrapper.api_request(
            'themes',
            f'fields *; where id = {theme_id_str};'
        )
        themes_response = json.loads(byte_array)
        themes = [r['name'] for r in themes_response]

        time.sleep(.25) # sleep to not go over request limit

        return themes