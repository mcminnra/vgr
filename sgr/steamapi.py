import json
import requests
import time
import xml.etree.ElementTree as ET

from lxml import html

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


def get_store_data(appid):
    # Get store html
    tree = get_steam_store_html(appid)

    # Init data
    data = {
        'name': None,
        'recent_percent': None,
        'recent_count': None,
        'all_percent': None,
        'all_count': None,
        'short_desc': None,
        'long_desc': None,
        'tags': None
    }

    try:
        data['name'] = get_name_from_html(tree)
        data['recent_percent'], data['recent_count'], data['all_percent'], data['all_count'] = get_reviews_from_html(tree)
        data['short_desc'] = get_short_desc_from_html(tree)
        data['long_desc'] = get_long_desc_from_html(tree)
        data['tags'] = get_tags_from_html(tree)
    except Exception as e:
        print(f'Failed pulling store data for {appid} - Does https://store.steampowered.com/app/{appid} exist?')

    return data