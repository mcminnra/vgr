from datetime import datetime
import json
import time
import xml.etree.ElementTree as ET

from lxml import html
import requests
from rich import print

WAIT_FOR_RESP_DOWNLOAD = 0.10


def get_library_appids():
    """
    Gets appids from steam library
    """

    r = requests.get('https://steamcommunity.com/id/ryder___/games?tab=all&xml=1')
    time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    root = ET.fromstring(r.text)[2]
  
    appids = []
    for game in root.findall('game'):
        appids.append(game.find('appID').text)

    return appids


def get_wishlist_appids():
    """
    Gets appids from steam wishlist
    """

    # Iterate through wishlist pages
    appids = []
    page_counter = 0
    while page_counter >= 0:
        r = requests.get(f'https://store.steampowered.com/wishlist/profiles/76561198053753111/wishlistdata/?p={page_counter}')
        time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    
        wishlist = json.loads(r.text)
        if wishlist:
            appids.append(list(wishlist.keys()))
            page_counter += 1
        else:
            page_counter = -1

    # Flatten list
    appids = [item for sublist in appids for item in sublist]

    return appids


def get_steam_store_html(appid):
    """
    Gets raw Steam store page HTML for a appid
    """
    r = requests.get(f'https://store.steampowered.com/app/{appid}')
    time.sleep(WAIT_FOR_RESP_DOWNLOAD)
    return html.fromstring(r.text)


def get_name_from_html(steam_store_tree):
    name = str(steam_store_tree.xpath('//div[@id="appHubAppName"]/text()')[0]).strip()
    return name


def get_release_date_from_html(steam_store_tree):
    date_list = steam_store_tree.xpath('//div[@class="date"]/text()')

    try:
        date_str = str(date_list[0]).strip()
        release_date = datetime.strptime(date_str, '%b %d, %Y').date()
    except Exception as e:
        release_date = None

    return release_date


def get_reviews_from_html(steam_store_tree):
    # Reviews
    reviews = [review.strip() for review in steam_store_tree.xpath('//span[@class="nonresponsive_hidden responsive_reviewdesc"]/text()') if '%' in review]
    
    # Remove some chars
    reviews = [r.replace(',', '').replace('%', '') for r in reviews]
    
    # Grab only numbers from reviews
    if len(reviews) == 1:
        #if no recent reviews, make recent the same as all
        recent_r = [int(s) for s in reviews[0].split() if s.isdigit()]
        all_r = [int(s) for s in reviews[0].split() if s.isdigit()]
    elif len(reviews) == 0:
        #if no reviews, set to 0
        recent_r = [0, 0]
        all_r = [0, 0]
    else: 
        recent_r = [int(s) for s in reviews[0].split() if s.isdigit()][:2]
        all_r = [int(s) for s in reviews[1].split() if s.isdigit()]

    return recent_r+all_r


def get_short_desc_from_html(steam_store_tree):
    desc_element = steam_store_tree.xpath('//div[@class="game_description_snippet"]/text()')

    short_desc = ""
    if desc_element:
        short_desc = str(desc_element[0]).strip()
        
    return short_desc


def get_long_desc_from_html(steam_store_tree):
    desc_element = steam_store_tree.xpath('//div[@id="game_area_description"]/text()')

    long_desc = ""
    if desc_element:
        long_desc = ''.join(desc_element).strip()
        
    return long_desc


def get_tags_from_html(steam_store_tree):
    """
    Gets the app tags from a Steam Store Page
    """
    tags = [tag.strip() for tag in steam_store_tree.xpath('//a[@class="app_tag"]/text()')]
    return tags

def get_is_dlc_from_html(steam_store_tree):
    """
    Gets dlc flag from Steam Store page
    """
    for item in steam_store_tree.xpath( "//div" ):
        if "game_area_dlc_bubble" in iter(item.classes):
            return True
    
    return False

def get_is_soundtrack_from_html(steam_store_tree):
    """
    Gets soundtrack flag from Steam Store page
    """
    for item in steam_store_tree.xpath( "//div" ):
        if "game_area_soundtrack_bubble" in iter(item.classes):
            return True
    
    return False

def get_is_video_from_html(steam_store_tree):
    """
    Gets video flag from Steam Store page
    """
    for item in steam_store_tree.xpath( "//span" ):
        if "streamingvideoseries" in iter(item.classes):
            return True
    
    return False


def get_store_data(appid):
    # Get store html
    # TODO Throw error on store page
    # TODO Think of a better way to do this
    tree = get_steam_store_html(appid)

    # Init data
    data = {
        'name': None,
        'release_date': None,
        'recent_percent': None,
        'recent_count': None,
        'all_percent': None,
        'all_count': None,
        'short_desc': None,
        'long_desc': None,
        'tags': None,
        'is_dlc': None,
        'is_soundtrack': None,
        'is_video': None,
    }

    try:
        data['name'] = get_name_from_html(tree)
    except Exception as e:
        print(f'Failed pulling store data for {appid} - Invalid AppID or AppID is unlisted from store - Does https://store.steampowered.com/app/{appid} or https://steamcommunity.com/app/{appid} exist?')
        return data
    
    try:
        data['release_date'] = get_release_date_from_html(tree)
        data['recent_percent'], data['recent_count'], data['all_percent'], data['all_count'] = get_reviews_from_html(tree)
        data['short_desc'] = get_short_desc_from_html(tree)
        data['long_desc'] = get_long_desc_from_html(tree)
        data['tags'] = get_tags_from_html(tree)
        data['is_dlc'] = get_is_dlc_from_html(tree)
        data['is_soundtrack'] = get_is_soundtrack_from_html(tree)
        data['is_video'] = get_is_video_from_html(tree)
    except Exception as e:
        print(f'Failed pulling store metadata for {appid} - Is https://store.steampowered.com/app/{appid} a weird store format?')

    return data
