import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import os
from help_functions import *


df_dir_experiments = os.path.join(os.getcwd(), 'data_frame_machineLearning.json')
df_dir_contacts = os.path.join(os.getcwd(), 'contacts_machineLearning.json')
if not os.path.isfile(df_dir_experiments) or not os.path.isfile(df_dir_contacts):
    raise ValueError('Currently the program does not have access to all necessary .json files.'
                     'Move data_frame_machineLearning.json and contacts_machineLearning.json to your current directory.')

df_experiments = pd.DataFrame(pd.read_json(df_dir_experiments))
df_winners = df_experiments[df_experiments['winner'] == True]

df_contacts = pd.read_json(df_dir_contacts).dropna()


def carrier_vs_path_length_linear_regression() -> None:
    """
    Is the number of carriers (ants attached to the load) linearly correlated to the path length of successful
    trajectories?
    :return:
    """
    title = 'Linear Regression for successful trials'
    xlabel, ylabel = 'average Carrier Number', 'path length/minimal path length[]'

    x = np.array(df_winners[xlabel]).reshape(-1, 1)
    y = np.array(df_winners[ylabel])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print_results('carrier_vs_path_length_linear_regression', model.score(x_test, y_test),
                  model.score(x_train, y_train))
    plot_model(model, x, y, xlabel, ylabel, title)


def velocity_vs_path_length_linear_regression() -> None:
    """
    Is the velocity of carrying linearly correlated to the path length of successful trajectories?
    :return:
    """
    title = 'Linear Regression for successful trials'
    xlabel, ylabel = 'velocity', 'path length/minimal path length[]'

    x = np.array(df_winners['velocity']).reshape(-1, 1)
    y = np.array(df_winners['path length/minimal path length[]'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print_results('velocity_vs_path_length_linear_regression', model.score(x_test, y_test),
                  model.score(x_train, y_train))
    plot_model(model, x, y, xlabel, ylabel, title)


def carrier_vs_successful_trials_linear_regression() -> None:
    """
    Is the number of carriers (ants attached to the load) linearly correlated to chance of completing the maze?
    :return:
    """
    title = 'LogisticRegression for successful trials'
    xlabel, ylabel = 'average Carrier Number', 'winner'

    x = np.array(df_experiments[[xlabel]]).reshape(-1, 1)
    y = np.array(df_experiments[ylabel])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    print_results('carrier_vs_successful_trials_linear_regression', model.score(x_test, y_test),
                  model.score(x_train, y_train))
    plot_model(model, x, y, xlabel, ylabel, title)


def torque_vs_rotation_at_contact() -> None:
    """
    At points, where the load bumps into a wall (so called 'contacts'): Does the rotation of the load match the applied
    torque?
    :return:
    """
    title = 'Linear Regression for torque vs. angular speed'
    xlabel, ylabel = 'torque', 'theta_dot'

    x = np.array(df_contacts[[xlabel]]).reshape(-1, 1)
    y = np.array(df_contacts[ylabel])
    x, y = remove_outliers(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print_results('torque_vs_rotation_at_contact', model.score(x_test, y_test), model.score(x_train, y_train))
    plot_model(model, x, y, xlabel, ylabel, title)


if __name__ == '__main__':
    carrier_vs_path_length_linear_regression()
    velocity_vs_path_length_linear_regression()
    carrier_vs_successful_trials_linear_regression()
    torque_vs_rotation_at_contact()
