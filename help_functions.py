import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


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


def remove_outliers(x, y):
    lof = LocalOutlierFactor(contamination=0.5, n_neighbors=20)
    yhat = lof.fit_predict(x)

    mask = yhat != -1
    non_mask = yhat == -1
    x, y = x[non_mask, :], y[non_mask]
    return x, y