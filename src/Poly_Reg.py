import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

F, N = map(int, input().split())
x, y = np.array([input().split() for _ in range(N)], float)
y=[]
for i in x:
	y.append(i[-1])
x = np.delete(x, 2, 1)
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(x)
y=np.array(y)
T = int(input())
X_test = np.array([input().split() for _ in range(T)], float)

X_test_ = poly.fit_transform(X_test)
# Instantiate
lg = LinearRegression()

# Fit
lg.fit(X_, y)

# Obtain coefficients
coeff=lg.coef_

# Predict
z=lg.predict(X_test_)
print(*z, sep='\n')