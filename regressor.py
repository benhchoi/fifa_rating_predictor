# TODO:
# 1. only care about the gk stats if the player is a gk
# 2. first condition on just skills
# 3. condition on position skills

import data_parser as dp

import os
import numpy as np
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


# separate the overall data into two parts:
# one that only contains goalkeepers
# another that contains every other type of player
def separate_by_gk(data):
    players = []
    gk = []

    for row in data:
        if row[dp.Data.POSITION] == 'GK':
            gk.append(row)
        else:
            players.append(row)

    return players, gk


# converts the data into X and y values
# this method decides which features to focus on in the regression
def extract_features(data, gk_only=False):
    y = [row[dp.Data.OVERALL] for row in data]

    if gk_only:
        X = [[  # row[dp.Data.POTENTIAL], row[dp.Data.REPUTATION],
            row[dp.Data.GK_DIVING], row[dp.Data.GK_HANDLING],
            row[dp.Data.GK_KICKING], row[dp.Data.GK_POSITIONING],
            row[dp.Data.GK_REFLEXES]]
            for row in data]

    return X, y


def get_gpr(X, y):
    kernel = kernels.ConstantKernel(1) * kernels.RBF(1)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.05)
    gpr.fit(X, y)

    return gpr


def generate_error_metrics(gpr, X, y):
    r_sq = gpr.score(X, y)

    y_pred = gpr.predict(X)
    mae = mean_absolute_error(y, y_pred)

    return r_sq, mae


def cross_validate(fold, repititions, X, y):
    r_sq_trains = []
    r_sq_tests = []
    mae_trains = []
    mae_tests = []

    for _ in range(repititions):
        kf = KFold(10)

        for train_index, test_index in kf.split(X):
            X_train = [X[index] for index in train_index]
            X_test = [X[index] for index in test_index]
            y_train = [y[index] for index in train_index]
            y_test = [y[index] for index in test_index]

            gpr = get_gpr(X_train, y_train)

            r_sq_train, mae_train = generate_error_metrics(
                gpr, X_train, y_train)
            r_sq_test, mae_test = generate_error_metrics(gpr, X_test, y_test)

            r_sq_trains.append(r_sq_train)
            r_sq_tests.append(r_sq_test)
            mae_trains.append(mae_train)
            mae_tests.append(mae_test)

    r_sq_train = np.mean(r_sq_trains)
    r_sq_test = np.mean(r_sq_tests)
    mae_train = np.mean(mae_trains)
    mae_test = np.mean(mae_tests)

    return r_sq_train, r_sq_test, mae_train, mae_test


def main():
    data = dp.get_csv(os.path.join('.', 'data.csv'))
    converted_data = dp.convert_data(data)

    players, gk = separate_by_gk(converted_data)

    X, y = extract_features(gk, True)

    r_sq_train, r_sq_test, mae_train, mae_test = cross_validate(10, 10, X, y)

    print('R squared train: %f, R squared test: %f' % (r_sq_train, r_sq_test))
    print('MAE train: %f, MAE test: %f' % (mae_train, mae_test))


if __name__ == '__main__':
    main()
