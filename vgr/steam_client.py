from functools import wraps
import json
import time
import xml.etree.ElementTree as ET

from lxml import html
import requests
from tenacity import retry, stop_after_attempt, wait_fixed


# Globals
WAIT_TIME = 0.1


class SteamClient():
    """API Client for Steam"""

    def __init__(self, url_name, user_id):
        self.url_name = url_name
        self.user_id = user_id

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_steam_library(self):
        """
        Gets (names, appids) from steam library
        """
        r = requests.get(f'https://steamcommunity.com/id/{self.url_name}/games?tab=all&xml=1', timeout=60)
        time.sleep(WAIT_TIME)
        root = ET.fromstring(r.text)[2]
    
        games = []
        for library_item in root.findall('game'):
            game = {}
            game['steam_id'] = int(library_item.find('appID').text)
            game['name'] = library_item.find('name').text
            games.append(game)

        return games

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_steam_wishlist(self):
        """
        Gets (names, appids) from steam wishlist
        """
        # Iterate through wishlist pages
        games = []
        page_counter = 0
        while page_counter >= 0:
            r = requests.get(f'https://store.steampowered.com/wishlist/profiles/{self.user_id}/wishlistdata/?p={page_counter}', timeout=60)
            time.sleep(WAIT_TIME)
        
            wishlist = json.loads(r.text)
            if wishlist:
                steam_ids = list(wishlist.keys())
                games += [{'steam_id': int(steam_id), 'name': wishlist[steam_id]['name']} for steam_id in steam_ids]
                page_counter += 1
            else:
                page_counter = -1

        return games

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_steam_search_page(self, steam_url):
        """
        Gets (names, appids) from a particular Steam Search Page
        """
        r = requests.get(steam_url, timeout=60)
        time.sleep(WAIT_TIME)
        tree = html.fromstring(r.text)
        
        games = []
        for link in tree.xpath("//a[@data-ds-appid]"):
            game = {
                'steam_id': int(link.get('data-ds-appid').split(',')[0]),
                'name': link.find_class("title")[0].text_content()
            }
            games.append(game)

        return games

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def get_steam_store_html(self, appid):
        """
        Gets raw Steam store page HTML for a appid
        """
        r = requests.get(f'https://store.steampowered.com/app/{appid}', timeout=60)
        time.sleep(WAIT_TIME)
        return html.fromstring(r.text)

    def get_name_from_html(self, steam_store_tree):
        return str(steam_store_tree.xpath('//div[@id="appHubAppName"]/text()')[0]).strip()

    def get_reviews_from_html(self, steam_store_tree):
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

    def get_short_desc_from_html(self, steam_store_tree):
        desc_element = steam_store_tree.xpath('//div[@class="game_description_snippet"]/text()')

        short_desc = ""
        if desc_element:
            short_desc = str(desc_element[0])
            
        return short_desc.strip().replace("\r", "").replace("\n", "")

    def get_tags_from_html(self, steam_store_tree):
        """
        Gets the app tags from a Steam Store Page
        # InitAppTagModal?
        """
        tags_raw = steam_store_tree.xpath('//a[@class="app_tag"]/text()')

        if tags_raw:
            tags = [tag.strip() for tag in tags_raw]
        else:
            tags = list()

        return tags