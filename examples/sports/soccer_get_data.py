"""Example how to get and store data"""

from ratingslibulf.datasets.filenames import download_and_store_footballdata
from ratingslibulf.datasets.soccer import championships

epl = championships.PREMIERLEAGUE  # ENGLISH PREMIER LEAGUE
download_and_store_footballdata(2005, 2018, championship=epl)
