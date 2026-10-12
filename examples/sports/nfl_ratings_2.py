"""NFL teams ratings with Keener"""


from ratingslibulf.datasets.filenames import (
    FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF, dataset_path)
from ratingslibulf.ratings.keener import Keener
from ratingslibulf.utils.methods import print_pandas

filename = dataset_path(FILENAME_NFL_2009_2010_SHORTNAMES_NO_PLAYOFF)
keener = Keener(normalization=False).rate_from_file(filename)
print_pandas(keener)
