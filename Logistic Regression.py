import numpy as np
import polars as pl
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
data = load_breast_cancer()
df = pl.DataFrame(data.data,schema=data.feature_names.tolist()).with_columns(pl.Series("target",data.target))

features = data.feature_names.tolist()
df = df.with_columns([
    (
        (pl.col(col) - pl.col(col).mean())/(pl.col(col).std())
    ).alias(col)
    for col in features
])

X = df.select(features).to_numpy()
y = df["target"].to_numpy()
print("="*20)
print("Logistic Regression from scratch")
print("="*20)
class logistic_regression:
    def __init__(self,learning_rate=0.1,epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    def fit(self,X,y):
        n =len(X)
        self.w = np.random.rand(X.shape[1])
        self.b = 0
        for epoch in range(self.epochs):
            z = X @ self.w + self.b
            pred = self.sigmoid(z)
            loss = -np.mean(y * np.log(pred + 1e-8) + (1-y) * np.log(1-pred + 1e-8))
            dz = pred - y
            dw = (X.T @ dz)/n
            db = np.mean(dz)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if epoch % 100 == 0:
                print("Epoch:",epoch,"Loss:",loss)
    def predict(self,X):
        z = X @ self.w + self.b
        prob = self.sigmoid(z)
        return (prob >= 0.5).astype(int)
model = logistic_regression()
model.fit(X,y)
pred = model.predict(X)

print("accuracy",np.mean(pred==y))


print("=" * 20)
print("SKlearn Logistic Regression")
print("=" * 20)

model = LogisticRegression()
model.fit(X,y)
pred = model.predict(X)
accuracy = np.mean(pred==y)
print("accuracy:",accuracy)

