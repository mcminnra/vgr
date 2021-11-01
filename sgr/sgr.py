#!/usr/bin/env python3

import argparse
import pathlib

import pandas as pd
from rich import print
from rich.console import Console

from etl import get_data

# CLI
parser = argparse.ArgumentParser(description='Steam Games Recommender')
parser.add_argument(
    'reviews_filepath',
    type=str,
    help='Reviews input filepath that has a "Steam AppID" and "Rating" columns.')

if __name__ == '__main__':
    # get print console
    console = Console()

    # Get args
    args = parser.parse_args()

    ### ETL    
    # Get reviewed games
    print(f'Reviews Input Path: {args.reviews_filepath}')
    df_scores = pd.read_excel(args.reviews_filepath)

    # Create data by getting library and wishlist games and enriching input
    df = get_data(df_scores)
    df.to_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/data.csv')  # cache
    del df

    ### Data Pre-processing
    df = pd.read_csv(str(pathlib.Path(__file__).parent.parent.absolute()) + '/data/data.csv')
    print(df)
