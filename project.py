import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
data = np.loadtxt(fname='diabetes.csv', delimiter=',', skiprows=1)
v = data[:, :8]
y = data[:, 8]
c = [v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4],v[:,5],v[:,6],v[:,7]]
diabetes_features = [x for i,x in enumerate(v) if i!=8]
'''for i in c:
    plt.scatter(i, y, color='g')
    plt.show()'''
X_train, X_test, y_train, y_test = train_test_split(v,y,test_size=0.2)

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
print("Feature importances:\n{}".format(tree.feature_importances_))
def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_diabetes(tree)
plt.savefig('feature_importance')
plt.show()
a=data[:, [1,5,7]]
trainx, testx, trainy, ytest = train_test_split(a,y ,test_size=0.2)
regressor = LinearRegression()
regressor.fit(trainx, trainy)
y_pred = regressor.predict(testx)
print('predicted values for x_test',y_pred)
plt.scatter(range(0, len(y_pred)),y_pred , color='blue', linewidth=3)
plt.scatter(range(0, len(trainy)), trainy, color='red', linewidth=3)
m_squared_error = mean_squared_error(ytest, y_pred)
print("Mean Squared Error " + str(m_squared_error))
plt.show()
trainx, testx, trainy, ytest = train_test_split(v,y ,test_size=0.2)
regressor = LinearRegression()
regressor.fit(trainx, trainy)
y_pred = regressor.predict(testx)
print('predicted values for x_test',y_pred)
plt.scatter(range(0, len(y_pred)),y_pred , color='blue', linewidth=3)
plt.scatter(range(0, len(trainy)), trainy, color='red', linewidth=3)
m_squared_error = mean_squared_error(ytest, y_pred)
print("Mean Squared Error " + str(m_squared_error))
plt.show()


