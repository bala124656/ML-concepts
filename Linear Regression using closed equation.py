import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([5,7,8,9,10])

x = np.concatenate((np.ones(len(x)), x)).reshape(-1,2)
theta = np.linalg.inv(x.T @ x) @ x.T @ y
print(theta)
b = theta[0]
w = theta[1]
print(b)
print(w)
prediction = x @ theta
print(prediction)
# test data
x_new = 6
y_pred = w * x_new + b
print(y_pred)
