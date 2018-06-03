import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


x = np.array([[0.44, 0.68], [0.99, 0.23], [0.84, 0.29], [0.28, 0.45], [0.07, 0.83], [0.66, 0.8], [0.73, 0.92],[0.57, 0.43], [0.43, 0.89], [0.27, 0.95], [0.43, 0.06], [0.87, 0.91], [0.78, 0.69], [0.9, 0.94], [0.41, 0.06]])

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(x)
X_test=np.array([[0.05,0.54],[0.91,0.91],[0.31,0.76],[0.51,0.31]])
X_test_ = poly.fit_transform(X_test)
y=np.array([511.14,717.1,607.91,270.4,289.88,830.85,1038.09,455.19,640.17,511.06,177.03,1242.52,891.37,1339.72,169.88])


# Instantiate
lg = LinearRegression()

# Fit
lg.fit(X_, y)

# Obtain coefficients
coeff=lg.coef_

# Predict
z=lg.predict(X_test_)
print(z)