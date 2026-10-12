"""Predictions based only on rating values"""

import ratingslibulf.ratings as rl
import pandas as pd
from ratingslibulf.app_sports.methods import Predictions, prepare_sports_seasons
from ratingslibulf.application import SoccerOutcome
from ratingslibulf.datasets.filenames import get_seasons_dict_footballdata_online
from ratingslibulf.datasets.soccer import championships
from ratingslibulf.datasets.parameters import stats
from ratingslibulf.utils.enums import ratings

pd.set_option('float_format', "{:.4f}".format)

filenames_dict = get_seasons_dict_footballdata_online(
    season_start=2009, season_end=2018, championship=championships.PREMIERLEAGUE)

ratings_list = [
    rl.Winloss(normalization=True),
    rl.Colley(),
    rl.Massey(data_limit=20),
    rl.Elo(version=ratings.ELOWIN, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Elo(version=ratings.ELOPOINT, K=40, ks=400, HA=0,
           starting_point=0),
    rl.Keener(normalization=True),
    rl.OffenseDefense(tol=0.0001),
    rl.Markov(b=0.85, stats_markov_dict=stats.STATS_MARKOV_DICT),
    rl.AccuRate()
]
outcome = SoccerOutcome()
data = prepare_sports_seasons(filenames_dict,
                              outcome,
                              rating_systems=ratings_list,
                              start_week=2)


results = Predictions(data, outcome, start_from_week=4).rs_pred_parallel(
    rating_systems=ratings_list,
    pred_methods_list=['MLE', 'RANK'])
