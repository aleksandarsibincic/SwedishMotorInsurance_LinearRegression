import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

data = pd.read_csv("./dataset/swedish_motor_insurance/SwedishMotorInsurance.csv", header=0)

description = data.describe()
print(description)

#This displays distplots for claims and payments
#We can see that the distributions have approximately the same shape which indicates that there is a strong relationship between the feature and label.

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

ax1.set_title('Distribution of feature X i.e. Number of Claims')
sns.distplot(data.Claims,bins=50,ax=ax1)

ax2.set_title('Distribution of label Y i.e. Total Payment for Corresponding claims')
sns.distplot(data.Payment,bins=50,ax=ax2)
plt.savefig(os.path.expanduser("./output/plots/SwedishMotorInsurance_Distribution_Sibincic.png"))
plt.show()

# This displays the scatter plot for Feature and Label and fits an approximate regression line for the same.

fig , (ax1) = plt.subplots(1,1,figsize=(10,4))

ax1.set_title('Scatter plot between feature and Label')
sns.regplot(data=data,x='Claims',y='Payment',ax=ax1)
plt.savefig(os.path.expanduser("./output/plots/SwedishMotorInsurance_RegressionLine_Sibincic.png"))
plt.show()

# Here we will train the Linear Regression model from scikit-learn and check the RMSE for the Training Data itself.

from sklearn import  metrics
from sklearn import linear_model
X = pd.DataFrame(data.Claims)
Y = data.Payment
regr = linear_model.LinearRegression()
regr.fit(X,Y)
Y_pred = regr.predict(X)
mse = metrics.mean_squared_error(Y_pred,Y)
print('RMSE for Training set : %f' % (np.sqrt(mse)))



from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/5, random_state = 0)

### Exporting training and test set
from sklearn import metrics

train_array = np.column_stack((X_Train, Y_Train))
np.savetxt(os.path.expanduser("./output/SwedishMotorInsurance_TrainingSet_Sibincic.csv"), train_array, delimiter=",")

test_array = np.column_stack((X_Test, Y_Test))
np.savetxt(os.path.expanduser("./output/SwedishMotorInsurance_TestSet_Sibincic.csv"), test_array, delimiter=",")

### Fitting Simple Linear Regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)


# Visualising the Training set results

plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Total payment for all claims vs no of claims  (Training Set)')
plt.xlabel('no of claims')
plt.ylabel('total payment for all claims')
plt.savefig(os.path.expanduser("./output/plots/SwedishMotorInsurance_TrainingSet_Sibincic.png"))
plt.show()

# Visualising the Test set results

plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Total payment for all claims vs no of claims  (Test Set)')
plt.xlabel('no of claims')
plt.ylabel('total payment for all claims')
plt.savefig(os.path.expanduser("./output/plots/SwedishMotorInsurance_TestSet_Sibincic.png"))
plt.show()

# Predicting the Test set result

Y_Pred = regressor.predict(X_Test)

#creating set with predictions

predictions_array = np.column_stack((X_Test, Y_Pred))
np.savetxt(os.path.expanduser("./output/SwedishMotorInsurance_PredictionSet_Sibincic.csv"), predictions_array, delimiter=",")

from sklearn.metrics import mean_squared_error, r2_score

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('RMSE: %.2f'
      % np.sqrt(mean_squared_error(Y_Test, Y_Pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_Test, Y_Pred))

# Plot outputs
plt.scatter(X_Test, Y_Test,  color='black')
plt.plot(X_Test, Y_Pred, color='blue', linewidth=3)
plt.title('Total payment for all claims vs no of claims  (Prediction)')
plt.xlabel('no of claims')
plt.ylabel('total payment for all claims')
plt.savefig(os.path.expanduser("./output/plots/SwedishMotorInsurance_Prediction_Sibincic.png"))
plt.show()

#Polynomial regression

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(3)

X_transform = poly.fit_transform(X_Train)
X_test_transf = poly.fit_transform(X_Test)


regressor.fit(X_transform,Y_Train) 

# Predicting the Test set result

Y_Pred = regressor.predict(X_test_transf)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print('RMSE: %.2f'
      % np.sqrt(mean_squared_error(Y_Test, Y_Pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_Test, Y_Pred))

#This needs to be reviewed

# Plot outputs
#plt.scatter(X_Test, Y_Test,  color='black')
#plt.plot(X_Test, Y_Pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()

#Ridge regression

from sklearn.linear_model import Ridge

rr = Ridge(alpha=100000)
rr.fit(X_Train, Y_Train) 

Y_Test_Pred= rr.predict(X_Test)

print('RMSE: %.2f'
      % np.sqrt(mean_squared_error(Y_Test, Y_Test_Pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_Test, Y_Test_Pred))

#Lasso regression

from sklearn.linear_model import Lasso

lr = Lasso(alpha=100000)
lr.fit(X_Train, Y_Train) 

Y_Test_Pred= lr.predict(X_Test)

print('RMSE: %.2f'
      % np.sqrt(mean_squared_error(Y_Test, Y_Test_Pred)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(Y_Test, Y_Test_Pred))

f= open(os.path.expanduser("./output/SwedishMotorInsurance_RMSE_Sibincic.txt"),"w+")
f.write(str(np.sqrt(mean_squared_error(Y_Test, Y_Test_Pred))))
f.close()
