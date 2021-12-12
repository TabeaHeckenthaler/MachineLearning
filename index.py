import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from help_functions import *


df_dir_experiments = os.path.join(os.getcwd(), 'data_frame_machineLearning.json')
df_experiments = pd.DataFrame(pd.read_json(df_dir_experiments))

df_dir_contacts = os.getcwd() + '\\contacts_machineLearning.json'
df_contacts = pd.read_json(df_dir_contacts).dropna()


# Carrier Number vs. Path Length Linear Regression Model
def carrier_vs_path_length_linear_regression():
    title = 'Linear Regression for successful trials'
    xlabel, ylabel = 'average Carrier Number', 'path length/minimal path length[]'

    x = np.array(df_experiments[xlabel]).reshape(-1, 1)
    y = np.array(df_experiments[ylabel])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('train coefficient of determination:', model.score(x_train, y_train))
    print('test coefficient of determination:', model.score(x_test, y_test))
    plot_model(model, x, y, xlabel, ylabel, title)


# Velocity vs. Path Length Linear Regression Model
def velocity_vs_path_length_linear_regression():
    df_winners = df_experiments[df_experiments['winner'] == True]
    title = 'Linear Regression for successful trials'
    xlabel, ylabel = 'velocity', 'path length/minimal path length[]'

    x = np.array(df_winners['velocity']).reshape(-1, 1)
    y = np.array(df_winners['path length/minimal path length[]'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('train coefficient of determination:', model.score(x_train, y_train))
    print('test coefficient of determination:', model.score(x_test, y_test))
    plot_model(model, x, y, xlabel, ylabel, title)


# Carrier numbers vs. successful trials  Regression Model
def carrier_vs_successful_trials_linear_regression():
    title = 'LogisticRegression for successful trials'
    xlabel, ylabel = 'average Carrier Number', 'winner'

    x = np.array(df_experiments[[xlabel]]).reshape(-1, 1)
    y = np.array(df_experiments[ylabel])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    print('train coefficient of determination:', model.score(x_train, y_train))
    print('test coefficient of determination:', model.score(x_test, y_test))
    plot_model(model, x, y, xlabel, ylabel, title)


# Contacts: Does the rotation match the torque?
def torque_vs_rotation_at_contact():
    title = 'Linear Regression for torque vs. angular speed'
    xlabel, ylabel = 'torque', 'theta_dot'

    x = np.array(df_contacts[[xlabel]]).reshape(-1, 1)
    y = np.array(df_contacts[ylabel])
    x, y = remove_outliers(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('train coefficient of determination:', model.score(x_train, y_train))
    print('test coefficient of determination:', model.score(x_test, y_test))
    plot_model(model, x, y, xlabel, ylabel, title)
    plt.show()


if __name__ == '__main__':
    carrier_vs_path_length_linear_regression()
    velocity_vs_path_length_linear_regression()
    carrier_vs_successful_trials_linear_regression()
    torque_vs_rotation_at_contact()