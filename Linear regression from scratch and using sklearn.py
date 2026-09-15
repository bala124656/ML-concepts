import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression as sklearnLinearRegression
from sklearn.metrics import mean_squared_error,r2_score


df = pl.read_csv("housing.csv")
X = df["median_income"].to_numpy()
Y = df["median_house_value"].to_numpy()

# covert ot 2d matrix
X = X.reshape(-1,1)

#we will create training and testing  data manually

indices = np.arange(len(X))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train, test = indices[:split], indices[split:]
X_train, y_train = X[train], Y[train]
X_test, y_test = X[test], Y[test]

class LinearRegression:
    def __init__(self,learning_rate=0.001,epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.loss_history = []

    def fit(self,X,y):
        n = len(X)
        self.w = np.zeros(X.shape[1])
        self.b = 0


        for epoch in range(self.epochs):
            y_pred =  X @ self.w   + self.b

            # residual
            error = y_pred - y
            #mse
            mse = np.mean(np.square(error))
            self.loss_history.append(mse)

            #Gradient
            dw  = 1/n * (X.T @ error)
            db = 1/n * np.sum(error)

            # update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
    def predict(self,X):
        return  X @ self.w  + self.b
    def loss(self,y_pred,y):
        return np.mean((y_pred - y)**2)
    def r2(self,y_pred,y_true):
        ss_res =np.sum(np.square(y_true - y_pred))
        ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
        return  1 - (ss_res/ss_tot)


    print( "=" * 20)
    print("model training ")
    print( "=" * 20)
model = LinearRegression(learning_rate=0.001,epochs=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = model.loss(y_test,y_pred)
r2 = model.r2(y_test,y_pred)

print("Weight:", model.w)
print("Bias:", model.b)
print("MSE:", mse)
print("R²:", r2)


sk_model = sklearnLinearRegression()
sk_model.fit(X_train,y_train)
sk_pred = sk_model.predict(X_test)


print("=" * 20)
print("sklearn model training ")
print( "=" * 20)
print("Weight:", sk_model.coef_)
print("Bias:", sk_model.intercept_)
print("Mse:",mean_squared_error(y_test,sk_pred))
print("R2:",r2_score(y_test,sk_pred))

