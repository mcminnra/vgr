import pandas as pd
from rich.progress import track

from steamapi import get_library_appids, get_wishlist_appids, get_store_data

def get_data(df_input):
    # Process scores df
    df = df_input[df_input['AppID'].notnull()]
    df = df.astype({'AppID': int})
    df = df[['AppID', 'Score']]
    print(f'Number of Reviewed Steam Games: {df.shape[0]}')

    # Get library appids
    appids_library = get_library_appids()
    df_library = pd.DataFrame({'AppID': appids_library, 'Score':None})
    df_library = df_library.astype({'AppID': int})
    print(f'Number of Steam Games Found in Library: {df_library.shape[0]}')

    # Get wishlist appids
    appids_wishlist = get_wishlist_appids()
    df_wishlist = pd.DataFrame({'AppID': appids_wishlist, 'Score':None})
    df_wishlist = df_wishlist.astype({'AppID': int})
    print(f'Number of Steam Games Found in Wishlist: {df_wishlist.shape[0]}')

    # Join appids
    df = df.merge(df_library, on='AppID', how='outer', suffixes=('', '_y'))[['AppID', 'Score']]
    df = df.merge(df_wishlist, on='AppID', how='outer', suffixes=('', '_y'))[['AppID', 'Score']]
    df = df.set_index('AppID')

    # Enrich appids
    for appid in track(df.index.values, description='Enriching with Steam store data'):
        data_appid = get_store_data(appid)
        keys = data_appid.keys()

        for key in keys:
            if key not in df.columns:
                df[key] = None
            df.at[appid, key] = data_appid[key]

    return df
