
import requests
from tqdm import tqdm
from igdb.wrapper import IGDBWrapper

import yaml
import pathlib

from game import Game
from steamapi import get_steam_library, get_steam_wishlist

# Load config
with open(str(pathlib.Path(__file__).parent.parent.absolute()) + '/config.yml', "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Get IGDB
igdb_client_id = "hlax4aqmbgg9ke7pu6ijaajemr92uk"
igdb_client_secret = "xpmve0632748x2vl4iu7srl80ecl5z"
r = requests.post(f"https://id.twitch.tv/oauth2/token?client_id={igdb_client_id}&client_secret={igdb_client_secret}&grant_type=client_credentials")
print(r.text)

igdb_client = IGDBWrapper("hlax4aqmbgg9ke7pu6ijaajemr92uk", "2jq81qzucxhfmgwbkkcdnvzj1nxu5w")
games = []

# # #Test games
# games = [
#     Game(igdb_client, name="Batman: Arkham Knight", igdb_id=5503),
#     Game(igdb_client, name="Batman: Arkham Knight", steam_id=208650),
# ]

print(get_steam_wishlist(config['steam_user_id']))

for item in tqdm(get_steam_library(config['steam_url_name'])[:10], desc='Getting Steam Library'):
    game = Game(igdb_client, steam_id=item['steam_id'], name=item['name'])
    if game not in games:
        games.append(game)

for game in games:
    print(game)
    print(game.igdb_metadata)
    print(game.steam_metadata)
    print()