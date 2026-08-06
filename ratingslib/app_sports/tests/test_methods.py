"""Module for test methods module"""

# Author: Kyriacos Talattinis <ktalattinis@gmail.com>
#
# Licence: MIT

import os
import unittest

from ratingslib.app_sports.methods import predict_hindsight
from ratingslib.application import SoccerOutcome
from ratingslib.datasets.filenames import FILENAME_EPL_2018_2019_20_GAMES
from ratingslib.datasets.parameters import COLUMNS_DICT
from ratingslib.datasets.parse import parse_pairs_data
from ratingslib.ratings.accurate import AccuRate
from ratingslib.tests.test_all import printdetails
from ratingslib.utils.methods import get_filename

current_dirname = os.path.dirname(__file__)
directory_path = r"../../datasets/"
FP_FILENAME_EPL_2018_2019_20_GAMES = get_filename(FILENAME_EPL_2018_2019_20_GAMES,
                                                  directory_path=directory_path,
                                                  current_dirname=current_dirname)


class TestMethods(unittest.TestCase):
    """
    A class to test functions from 
    :mod:`ratingslib.app_sports.methods` 
    module The results are based on the filename (FILENAME_EPL_2018_2019_20_GAMES)
    which contains the first two match-weeks of the English Premier League
    soccer championship during the season 2018-2019
    """
    @printdetails
    def test_calc_predictions(self):
        """
        Test the prediction procedure according to rating values.
        The logic of prediction is that a higher rating is preferred than
        the lower rating.
        For example in a match between teamA and teamB
        with ratingA and ratingB respectively,
        if ratingA > ratingB then prediction of the winner is teamA
        if ratingA < ratingB then prediction of the winner is teamB
        if ratingA = ratingB then prediction is the Draw.
        """
        columns_dict = COLUMNS_DICT
        outcome = SoccerOutcome()
        data, teams_df = parse_pairs_data(FP_FILENAME_EPL_2018_2019_20_GAMES,
                                          outcome=outcome,
                                          columns_dict=columns_dict)
        ac = AccuRate().rate(data, teams_df, columns_dict=columns_dict)
        # hindsight
        pred, _ = predict_hindsight(
            data, ac, outcome, columns_dict=columns_dict)
        self.assertListEqual(pred,
                             [2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2])

    @printdetails
    def test_foresight_unplayed_matches_week_collision(self):
        import numpy as np
        import pandas as pd
        from ratingslib.app_sports.methods import prepare_sport_dataset, Predictions
        from ratingslib.ratings.elo import Elo
        columns_dict = COLUMNS_DICT.copy()
        outcome = SoccerOutcome()
        data_train, teams_df = parse_pairs_data(FP_FILENAME_EPL_2018_2019_20_GAMES,
                                                outcome=outcome,
                                                columns_dict=columns_dict)
        data_train['Week_Number'] = np.arange(1, len(data_train) + 1)
        
        data_test = data_train.iloc[:5].copy()
        data_test['FTHG'] = np.nan
        data_test['FTAG'] = np.nan
        data_test[outcome.name] = np.nan
        data_test['Week_Number'] = 18
        
        data_all = pd.concat([data_train, data_test]).reset_index(drop=True)
        ac = AccuRate()
        data_prep = prepare_sport_dataset(data_all, teams_df, ac, start_week=2, columns_dict=columns_dict)
        
        preds = Predictions(data_prep, outcome, start_from_week=18,
                            print_accuracy_report=True,
                            print_predictions=True,
                            print_classification_report=True,
                            columns_dict=columns_dict)
        test_Y, pred = preds.rs_pred(pred_method='MLE', ratings=ac)
        self.assertEqual(len(pred), 5)
        
        # Test parallel execution with report printing on unplayed matches (NaN actual outcomes)
        results_par = preds.rs_pred_parallel(rating_systems={'AccuRate': ac}, pred_methods_list=['MLE'])
        self.assertIn('MLE', results_par)
        self.assertEqual(len(results_par['MLE']['AccuRATE'][1]), 5)

    @printdetails
    def test_prepare_forecast_dataset(self):
        import numpy as np
        import pandas as pd
        from ratingslib.app_sports.methods import prepare_forecast_dataset, prepare_sport_dataset, Predictions
        from ratingslib.ratings.elo import Elo
        columns_dict = COLUMNS_DICT.copy()
        outcome = SoccerOutcome()
        data_train, teams_df = parse_pairs_data(FP_FILENAME_EPL_2018_2019_20_GAMES,
                                                outcome=outcome,
                                                columns_dict=columns_dict)
        data_train['Week_Number'] = np.arange(1, len(data_train) + 1)

        # Create separate upcoming matchday dataset (no concatenation required)
        data_test = data_train.iloc[:3].copy()
        data_test['FTHG'] = np.nan
        data_test['FTAG'] = np.nan
        data_test[outcome.name] = np.nan
        data_test['Week_Number'] = 100

        ac = AccuRate()
        train_prep, test_prep = prepare_forecast_dataset(data_train, data_test, teams_df, {'AccuRATE': ac}, start_week=2, columns_dict=columns_dict)
        self.assertIn('HratingnormAccuRATE', test_prep.columns)
        self.assertIn('AratingnormAccuRATE', test_prep.columns)

        # Also test via prepare_sport_dataset forwarding
        train_prep2, test_prep2 = prepare_sport_dataset(data_train, teams_df, {'AccuRATE': ac}, start_week=2, columns_dict=columns_dict, data_test=data_test)
        self.assertEqual(len(test_prep2), 3)

        preds = Predictions(train_prep, outcome, data_test=test_prep,
                            print_accuracy_report=True,
                            print_predictions=True,
                            print_classification_report=True,
                            columns_dict=columns_dict)
        results = preds.rs_pred_parallel(rating_systems={'AccuRATE': ac}, pred_methods_list=['MLE', 'RANK'])
        self.assertIn('MLE', results)
        self.assertIn('RANK', results)
        self.assertEqual(len(results['MLE']['AccuRATE'][1]), 3)
        self.assertEqual(len(results['RANK']['AccuRATE'][1]), 3)


if __name__ == '__main__':
    unittest.main()
