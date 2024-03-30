import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

table = pd.read_csv("classes.csv")

Y = np.array(table['Star type'])
X = np.array(table[['Absolute magnitude(Mv)']])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

regr = MLPRegressor(random_state=1, max_iter=10000)
regr.fit(X_train, Y_train)

print(regr.score(X_test, Y_test))

# corr = table.corr()
# sns.heatmap(corr, annot=True)
# plt.show()