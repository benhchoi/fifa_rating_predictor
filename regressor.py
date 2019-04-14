import numpy as np
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, median_absolute_error

from data_parser import Data


# separate the overall data into two parts:
# one that only contains goalkeepers
# another that contains every other type of player
def separate_by_gk(data):
    players = []
    gk = []

    for row in data:
        if row[Data.POSITION] == 'GK':
            gk.append(row)
        else:
            players.append(row)

    return players, gk


# takes all of the non-gk players
# separates them by forwards, midfielders, and defenders
def separate_players_by_3(players):
    forward_letters = {'W', 'F', 'S'}
    midfielder_letters = {'M'}
    defender_letters = {'B'}

    forwards = []
    midfielders = []
    defenders = []

    for row in players:
        position = row[Data.POSITION]
        if check_letters(forward_letters, position):
            forwards.append(row)
        elif check_letters(midfielder_letters, position):
            midfielders.append(row)
        elif check_letters(defender_letters, position):
            defenders.append(row)

    return forwards, midfielders, defenders


# checks if any of the letters in the given set of letters is in the word
def check_letters(letters, word):
    for letter in letters:
        if letter in word:
            return True

    return False


# converts the data into X and y values
# this method decides which features to focus on in the regression
# 5 extract_types: gk, players, forwards, midfielders, defenders
def extract_features(data, extract_type):
    y = [row[Data.OVERALL] for row in data]

    # only focus on goalkeeping
    gk_features = range(83, 88)

    # focus on everything but goalkeeping
    # pace, shooting, passing, dribbling, defending, physical
    player_features = range(54, 83)

    # focus on pace, shooting, passing, dribbling
    forward_features = [Data.ACCEL, Data.SPRINT_SPEED, Data.FINISHING,
                        Data.LONG_SHOTS, Data.PENALTIES, Data.POSITIONING,
                        Data.SHOT_POWER, Data.VOLLEYS, Data.CROSSING,
                        Data.CURVE, Data.FK_ACC, Data.LONG_PASS,
                        Data.SHORT_PASS, Data.VISION, Data.AGILITY,
                        Data.BALANCE, Data.BALL_CONTROL, Data.COMPOSURE,
                        Data.DRIBBLING, Data.REACTIONS]

    # focus on everything but goalkeeping
    midfielder_features = player_features

    # focus on pace, passing, defending, and physical
    defender_features = [Data.ACCEL, Data.SPRINT_SPEED, Data.CROSSING,
                         Data.CURVE, Data.FK_ACC, Data.LONG_PASS,
                         Data.SHORT_PASS, Data.VISION, Data.HEADING_ACC,
                         Data.INTERCEPTIONS, Data.MARKING, Data.SLIDE_TACKLE,
                         Data.STAND_TACKLE, Data.AGGRESSION, Data.JUMPING,
                         Data.STAMINA, Data.STRENGTH]

    X = []
    for row in data:
        if extract_type == 'gk':
            row_data = [row[i] for i in gk_features]
        elif extract_type == 'players':
            row_data = [row[i] for i in player_features]
        elif extract_type == 'forwards':
            row_data = [row[i] for i in forward_features]
        elif extract_type == 'midfielders':
            row_data = [row[i] for i in midfielder_features]
        elif extract_type == 'defenders':
            row_data = [row[i] for i in defender_features]
        X.append(row_data)

    return X, y


# creates a gp with the specified kernel fitted to the given X and y values
def get_gpr(kernel_type, X, y):
    mean, _, std = get_distribution_measures(y)
    if kernel_type == 'rbf':
        kernel = kernels.ConstantKernel(mean) * kernels.RBF(std)
    elif kernel_type == 'dot':
        kernel = kernels.ConstantKernel(mean) * kernels.DotProduct(std)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.05, optimizer=None)
    gpr.fit(X, y)

    return gpr


# gets the mean absolute error and the median absolute error of the gp
# predicted on the given X and y values
def get_errors(gpr, X, y):
    y_pred = gpr.predict(X)
    mean_error = mean_absolute_error(y, y_pred)
    median_error = median_absolute_error(y, y_pred)

    return mean_error, median_error


# runs a cross validation given a number of folds, X, and y values
# using a gp made with the specified kernel_type
def cross_validate(fold, kernel_type, X, y):
    mean_trains = []
    mean_tests = []
    median_trains = []
    median_tests = []
    kf = KFold(10)

    for train_index, test_index in kf.split(X):
        X_train = [X[index] for index in train_index]
        X_test = [X[index] for index in test_index]
        y_train = [y[index] for index in train_index]
        y_test = [y[index] for index in test_index]

        gpr = get_gpr(kernel_type, X_train, y_train)

        mean_train, median_train = get_errors(gpr, X_train, y_train)
        mean_test, median_test = get_errors(gpr, X_test, y_test)

        mean_trains.append(mean_train)
        mean_tests.append(mean_test)
        median_trains.append(median_train)
        median_tests.append(median_test)

    mean_train = np.mean(mean_trains)
    mean_test = np.mean(mean_tests)
    median_train = np.mean(median_trains)
    median_test = np.median(median_tests)

    return mean_train, mean_test, median_train, median_test


# exmaines poor predictions (>= e error) of gp trained on X and y
# using a trained_gp4
def examine_bad_predictions(trained_gpr, X, y, e):
    y_pred = trained_gpr.predict(X)

    poor_predictions = {}
    for i, pred in enumerate(y_pred):
        if abs(y[i] - pred) >= e:
            poor_predictions[i] = [pred, y[i]]

    return poor_predictions


# gets the mean, median, and standard deviation of an array of numbers
def get_distribution_measures(arr):
    return np.mean(arr), np.median(arr), np.std(arr)
