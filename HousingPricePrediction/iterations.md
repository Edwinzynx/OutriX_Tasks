## **making changes to the actual data, doesnt make it suitable for multiple code runs**

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model\_selection import train\_test\_split

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import mean\_squared\_error, r2\_score



\# Load original data from CSV (raw, untouched)

data = pd.read\_csv('housing.csv')



\# -------------------- Data Preprocessing -------------------- #



\# Convert yes/no to 1/0 only if not already numeric

binary\_cols = \['guestroom', 'mainroad', 'basement', 'hotwaterheating', 'prefarea', 'airconditioning']

for col in binary\_cols:

&nbsp;   if data\[col].dtype == 'object' or not pd.api.types.is\_numeric\_dtype(data\[col]):

&nbsp;       data\[col] = data\[col].astype(str).str.strip().str.lower()

&nbsp;       data\[col] = data\[col].map({'yes': 1, 'no': 0})



\# Map furnishingstatus only if not already numeric

if 'furnishingstatus' in data.columns:

&nbsp;   if data\['furnishingstatus'].dtype == 'object':

&nbsp;       data\['furnishingstatus'] = data\['furnishingstatus'].str.strip().str.lower()

&nbsp;       data\['furnishingstatus'] = data\['furnishingstatus'].map({

&nbsp;           'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0

&nbsp;       })



\# Create 'rooms' feature only if not present or has NaNs

if 'rooms' not in data.columns or data\['rooms'].isnull().any():

&nbsp;   data\['rooms'] = (

&nbsp;       data\['bedrooms'] \* 1.0 +

&nbsp;       data\['bathrooms'] \* 0.5 +

&nbsp;       data\['guestroom'] \* 1.0

&nbsp;   )



\# -------------------- Model Training -------------------- #



\# Features and label

X = data\[\['area', 'rooms', 'stories', 'mainroad', 'parking', 'prefarea', 'airconditioning']]

y = data\['price']



\# Train-test split

X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&nbsp;   X, y, test\_size=0.2, random\_state=42

)



\# Train linear regression model

model = LinearRegression()

model.fit(X\_train, y\_train)



\# Predict on test set

y\_pred = model.predict(X\_test)



\# Evaluation

mse = mean\_squared\_error(y\_test, y\_pred)

r2 = r2\_score(y\_test, y\_pred)

rmse = np.sqrt(mse)



print("Mean Squared Error (MSE):", mse)

print("Root Mean Squared Error (RMSE):", rmse)

print("R² Score (Coefficient of Determination):", r2)



\# -------------------- Visualization -------------------- #



plt.scatter(y\_test, y\_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.plot(\[y\_test.min(), y\_test.max()], \[y\_test.min(), y\_test.max()], color='red')  # reference line

plt.grid()

plt.show()



## **Optimised python script with LR and RFR**

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model\_selection import train\_test\_split

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import mean\_squared\_error, r2\_score

from sklearn.ensemble import RandomForestRegressor





\# Load original data from CSV (raw, untouched)

data = pd.read\_csv('housing.csv')



\# -------------------- Data Preprocessing -------------------- #



\# Convert yes/no to 1/0 only if not already numeric

binary\_cols = \['guestroom', 'mainroad', 'basement', 'hotwaterheating', 'prefarea', 'airconditioning']

for col in binary\_cols:

&nbsp;   if data\[col].dtype == 'object' or not pd.api.types.is\_numeric\_dtype(data\[col]):

&nbsp;       data\[col] = data\[col].astype(str).str.strip().str.lower()

&nbsp;       data\[col] = data\[col].map({'yes': 1, 'no': 0})



\# Map furnishingstatus only if not already numeric

if 'furnishingstatus' in data.columns:

&nbsp;   if data\['furnishingstatus'].dtype == 'object':

&nbsp;       data\['furnishingstatus'] = data\['furnishingstatus'].str.strip().str.lower()

&nbsp;       data\['furnishingstatus'] = data\['furnishingstatus'].map({

&nbsp;           'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0

&nbsp;       })



\# Create 'rooms' feature only if not present or has NaNs

if 'rooms' not in data.columns or data\['rooms'].isnull().any():

&nbsp;   data\['rooms'] = (

&nbsp;       data\['bedrooms'] \* 1.0 +

&nbsp;       data\['bathrooms'] \* 0.5 +

&nbsp;       data\['guestroom'] \* 1.0

&nbsp;   )

\#create area x stories column

if 'arsto' not in data.columns or data\['arsto'].isnull().any():

&nbsp;   data\['arsto'] = (

&nbsp;       data\['area'] \* data\['stories']

&nbsp;   )



\#create features column

if 'features' not in data.columns or data\['features'].isnull().any():

&nbsp;   data\['features'] = (data\['basement'] + data\['hotwaterheating'] + data\['airconditioning'] + data\['mainroad'] + data\['furnishingstatus'] )/(5)



\# -------------------- Model Training -------------------- #



\# Features and label

X = data\[\['area','features', 'arsto', 'bedrooms', 'bathrooms','stories', 'mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning',  'parking', 'prefarea','furnishingstatus', 'rooms']]

y = data\['price']



\# Train-test split

X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&nbsp;   X, y, test\_size=0.2, random\_state=42

)



\# Train linear regression model

model = LinearRegression()

model.fit(X\_train, y\_train)



\# Predict on test set

y\_pred = model.predict(X\_test)



\# Evaluation

mse = mean\_squared\_error(y\_test, y\_pred)

r2 = r2\_score(y\_test, y\_pred)

rmse = np.sqrt(mse)



print("Mean Squared Error (MSE):", mse)

print("Root Mean Squared Error (RMSE):", rmse)

print("R² Score (Coefficient of Determination):", r2)



\# -------------------- Visualization -------------------- #



fit = np.polyfit(y\_test, y\_pred, deg=1)

fit\_fn = np.poly1d(fit)

plt.plot(y\_test, fit\_fn(y\_test), color='green', linestyle='--', label='Prediction Trend')



plt.scatter(y\_test, y\_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices(Linear Regression)")

plt.plot(\[y\_test.min(), y\_test.max()], \[y\_test.min(), y\_test.max()], color='red')  # reference line

plt.grid()

plt.show()



\# -------------------- RandomForestRegression -------------------- #



model = RandomForestRegressor(

&nbsp;   n\_estimators=300,

&nbsp;   max\_depth=20,

&nbsp;   min\_samples\_split=10,

&nbsp;   min\_samples\_leaf=2,

&nbsp;   max\_features='sqrt',

&nbsp;   bootstrap=True,

&nbsp;   random\_state=42

)

model.fit(X\_train, y\_train)

\#testing on the test data

y\_pred = model.predict(X\_test)



\# ----------------------- Visualization ----------------------- #



fit = np.polyfit(y\_test, y\_pred, deg=1)

fit\_fn = np.poly1d(fit)

plt.plot(y\_test, fit\_fn(y\_test), color='green', linestyle='--', label='Prediction Trend')



plt.scatter(y\_test, y\_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices (Random Forest)")

plt.plot(\[y\_test.min(), y\_test.max()], \[y\_test.min(), y\_test.max()], color='red')  # perfect prediction line

plt.grid()

plt.show()



\# Evaluation for rfr

mserfr = mean\_squared\_error(y\_test, y\_pred)

r2rfr = r2\_score(y\_test, y\_pred)

rmserfr = np.sqrt(mserfr)



print("Mean Squared Error (MSE):", mserfr)

print("Root Mean Squared Error (RMSE):", rmserfr)

print("R² Score (Coefficient of Determination):", r2rfr)



\# Train Linear Regression

lr = LinearRegression()

lr.fit(X\_train, y\_train)

y\_pred\_lr = lr.predict(X\_test)



\# Train Random Forest Regressor

rfr = RandomForestRegressor(n\_estimators=100, random\_state=42)

rfr.fit(X\_train, y\_train)

y\_pred\_rfr = rfr.predict(X\_test)



\# Plot: Actual vs Predicted

plt.figure(figsize=(10, 6))



\# Scatter: Linear Regression

plt.scatter(y\_test, y\_pred\_lr, alpha=0.6, label='Linear Regression Predictions', color='blue')



\# Scatter: Random Forest

plt.scatter(y\_test, y\_pred\_rfr, alpha=0.6, label='Random Forest Predictions', color='green')



\# Reference line (Perfect Prediction: 45 degrees)

plt.plot(\[y\_test.min(), y\_test.max()], \[y\_test.min(), y\_test.max()], color='red', linewidth=2, label='Perfect Prediction (45°)')



\# Trendline: Linear Regression

lr\_fit = np.poly1d(np.polyfit(y\_test, y\_pred\_lr, deg=1))

plt.plot(y\_test, lr\_fit(y\_test), color='blue', linestyle='--', label='LR Trendline')



\# Trendline: Random Forest

rfr\_fit = np.poly1d(np.polyfit(y\_test, y\_pred\_rfr, deg=1))

plt.plot(y\_test, rfr\_fit(y\_test), color='green', linestyle='--', label='RFR Trendline')



\# Labels and Legend

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Model Comparison: Actual vs Predicted")

plt.legend()

plt.grid(True)

plt.tight\_layout()

plt.show()



