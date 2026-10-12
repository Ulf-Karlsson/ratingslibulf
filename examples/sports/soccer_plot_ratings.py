"""Plot rating values of teams"""

import ratingslibulf.ratings as rl
import pandas as pd
from ratingslibulf.app_sports.methods import prepare_sports_seasons
from ratingslibulf.application import SoccerOutcome
from ratingslibulf.datasets.filenames import get_seasons_dict_footballdata_online
from ratingslibulf.datasets.soccer import championships
from ratingslibulf.datasets.parameters import COLUMNS_DICT
from ratingslibulf.ratings.methods import plot_ratings
from ratingslibulf.utils.enums import ratings
from ratingslibulf.utils.methods import parse_columns

pd.set_option('float_format', "{:.4f}".format)

# 2009-2010 EPL season
filenames_dict = get_seasons_dict_footballdata_online(
    season_start=2009, season_end=2010, championship=championships.PREMIERLEAGUE)

elo = rl.Elo(version=ratings.ELOWIN, K=40, ks=400, HA=0,
             starting_point=1500)

columns_dict = COLUMNS_DICT
outcome = SoccerOutcome()
data_df = prepare_sports_seasons(filenames_dict,
                                 outcome,
                                 rating_systems=elo,
                                 start_week=1,
                                 preprocess=None,
                                 columns_dict=columns_dict)

col_names = parse_columns(columns_dict)
# plot ratings of Arsenal and Liverpool for 2009-2010 season
plot_ratings(data_df[2009], col_names.item_i, col_names.item_j,
             'H'+elo.params_key, 'A'+elo.params_key,
             starting_game=1, items_list=['Arsenal', 'Liverpool'])
