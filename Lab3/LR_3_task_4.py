import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= 0.5, random_state = 0)

regr = linear_model.LinearRegression()

regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Regr coef: " + str(regr.coef_))
print("Regr intercept: " + str(regr.intercept_))

print("R2 score: " + str(r2_score(ytest,ypred)))
print("Mean absolute error: " + str(mean_absolute_error(ytest,ypred)))
print("Mean squared error: " + str(mean_squared_error(ytest,ypred)))


fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Measured')
ax.set_ylabel('Provided')
plt.show()