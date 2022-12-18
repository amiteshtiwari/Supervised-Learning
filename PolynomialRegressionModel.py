#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# creating features and response
np.random.seed(1)
x_1 = np.absolute(np.random.randn(100, 1)*10)
x_2 = np.absolute(np.random.randn(100, 1)*30)
y = 2*x_1**2 + 3*x_1 + 2 + np.random.randn(100, 1)*20
# depicting the graph
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].scatter(x_1, y)
axes[1].scatter(x_2, y)
axes[0].set_title("x_1 plotted")
axes[1].set_title("x_2 plotted")
#plt.show()
# stored variables in a data frame
df = pd.DataFrame({"x_1":x_1.reshape(100,), "x_2":x_2.reshape(100,),
                          "y":y.reshape(100,)}, index=range(0,100))
print(df.loc[1])
# define train and test data
from sklearn.model_selection import train_test_split
X, y = df[["x_1", "x_2"]], df["y"]
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X.values)
X_train, X_test, y_train, y_test = train_test_split(poly_features,
                               y, test_size=0.4, random_state=42)
# creating a polynomial regression model
from sklearn.linear_model import LinearRegression
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)
# testing the model
poly_reg_y_predicted = poly_reg_model.predict(X_test)
from sklearn.metrics import mean_squared_error
poly_reg_rmse = np.sqrt(mean_squared_error(y_test,
                                    poly_reg_y_predicted))
print (f"The value of root mean square  {poly_reg_rmse}")