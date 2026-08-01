<img src="docs/source/images/RatingsLib.png" width="50%" />

# RatingsLib: A python library for rating methods with applications

RatingsLib is a Python library dedicated to rating/ranking systems implementation 
with applications in sports and other fields. 
This is a fork of ktalattinis excelent project.
The repository's dependency management has been modernized to use a pyproject.toml following PEP 621.
## Installation

RatingsLib requires Python 3.8 or newer. 
Option 1: Install Directly from GitHub (Quickest)
If you just want to install the library to use it in your Python scripts on another PC, you don't even need to download the files manually. Just run this command:

powershell


pip install git+https://github.com/Ulf-Karlsson/ratingslibulf.git
This will automatically download the package, read the pyproject.toml, and install ratingslib and all its dependencies (Pandas, SciPy, etc.) into your environment.

Option 2: Clone and Install (For Development)
If you plan to modify the code on the other PC, you should clone the repository first and then install it:

powershell


# 1. Clone the repository
git clone https://github.com/Ulf-Karlsson/ratingslibulf.git
# 2. Enter the directory
cd ratingslibulf
# 3. Install the package
pip install .

```
## Implementation
Rating/Ranking systems:
 * WinLoss
 * Colley
 * Massey
 * Keener
 * Elo
 * Offense - Defense
 * GeM
 * AccuRATE

Ranking Aggregation methods:
 * Borda Count
 * Average Rank

Rating Aggregation methods:
 * Markov
 * Perron
 * Offense-Defense

Comparison metrics:
 * Kendall's Tau

Applications & Examples:
   * Sports (the main application of the library):
      * Soccer Teams rating
      * Soccer Teams ranking lists comparison
      * Hindsight and foresight prediction of the final outcome of soccer matches
      * Combining rating systems and machine learning methods to predict soccer matches outcome
      * Ranking NFL teams

   * Other Applications & Examples:
      * Finance:
        * Examples from investment selection and portfolios rating and ranking.
      * Domain Market:
        * An illustrative example is provided and shows the ranking of domain names.
      * Movies:
        * Application on real-world dataset from [MovieLens](https://grouplens.org/datasets/movielens/)

## Documentation
The documentation is available at: https://ktalattinis.github.io/ratingslib/





