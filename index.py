import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import getcwd


df_dir = os.path.join(getcwd(), 'data_frame_machineLearning.json')
df = pd.DataFrame(pd.read_json(df_dir))
df_winners = df[df['winner'] == True]


def plot_model(model, x, y, xlabel, ylabel, title):
    x1, x2 = min(x)[0], max(x)[0]  # 0, size-1
    y1, y2 = model.predict([[x1], [x2]])
    plt.plot([x1, x2], [y1, y2], color="red")
    plt.scatter(x, y)
    # plt.xlim(0, x2)
    # plt.ylim(0, 60)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return


# Carrier Number vs. Path Length Linear Regression Model
def carrier_vs_path_length_linear_regression():
    title = 'Linear Regression for successful trials'
    xlabel, ylabel = 'average Carrier Number', 'path length/minimal path length[]'

    x = np.array(df[xlabel]).reshape(-1, 1)
    y = np.array(df[ylabel])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('train coefficient of determination:', model.score(x_train, y_train))
    print('test coefficient of determination:', model.score(x_test, y_test))
    plot_model(model, x, y, xlabel, ylabel, title)


# Velocity vs. Path Length Linear Regression Model
def velocity_vs_path_length_linear_regression():
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

    x = np.array(df[[xlabel]]).reshape(-1, 1)
    y = np.array(df[ylabel])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    print('train coefficient of determination:', model.score(x_train, y_train))
    print('test coefficient of determination:', model.score(x_test, y_test))
    plot_model(model, x, y, xlabel, ylabel, title)


def remove_outliers(x, y):
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(contamination=0.5, n_neighbors=20)
    yhat = lof.fit_predict(x)

    mask = yhat != -1
    non_mask = yhat == -1
    x, y = x[non_mask, :], y[non_mask]
    return x, y


# Contacts
def torque_vs_rotation_at_contact():
    df_dir = getcwd() + '\\contacts_machineLearning.json'
    df_contacts = pd.read_json(df_dir).dropna()

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

