import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


#Load original data from CSV
data = pd.read_csv('housing.csv')

#Data Preprocessing
#Convert yes/no to 1/0 only if not already numeric
binary_cols = ['guestroom', 'mainroad', 'basement', 'hotwaterheating', 'prefarea', 'airconditioning']
for col in binary_cols:
    if data[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].astype(str).str.strip().str.lower()
        data[col] = data[col].map({'yes': 1, 'no': 0})

#Map furnishingstatus only if not already numeric
if 'furnishingstatus' in data.columns:
    if data['furnishingstatus'].dtype == 'object':
        data['furnishingstatus'] = data['furnishingstatus'].str.strip().str.lower()
        data['furnishingstatus'] = data['furnishingstatus'].map({
            'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0
        })

#create rooms' feature only if not present or has NaNs
if 'rooms' not in data.columns or data['rooms'].isnull().any():
    data['rooms'] = (
        data['bedrooms'] * 1.0 +
        data['bathrooms'] * 0.5 +
        data['guestroom'] * 1.0
    )
#create area x stories column
if 'arsto' not in data.columns or data['arsto'].isnull().any():
    data['arsto'] = (
        data['area'] * data['stories']
    )

#create features column
if 'features' not in data.columns or data['features'].isnull().any():
    data['features'] = (data['basement'] + data['hotwaterheating'] + data['airconditioning'] + data['mainroad'] + data['furnishingstatus'] )/(5)

#Model Training

X = data[['area', 'features', 'arsto', 'bedrooms', 'bathrooms', 'stories',
          'mainroad', 'guestroom', 'basement', 'hotwaterheating',
          'airconditioning', 'parking', 'prefarea', 'furnishingstatus', 'rooms']]
y = data['price']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

#Evaluation for LR
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(" Linear Regression:")
print("  MSE:", mse_lr)
print("  RMSE:", rmse_lr)
print("  R² Score:", r2_lr)

#Random Forest Regressor
rfr = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)

#Evaluation of Random Forest
mse_rfr = mean_squared_error(y_test, y_pred_rfr)
rmse_rfr = np.sqrt(mse_rfr)
r2_rfr = r2_score(y_test, y_pred_rfr)

print("\n Random Forest Regressor:")
print("  MSE:", mse_rfr)
print("  RMSE:", rmse_rfr)
print("  R² Score:", r2_rfr)

#Comparison Plot
plt.figure(figsize=(10, 6))

# Scatter: Linear Regression
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='blue', label='Linear Regression')

# Scatter: Random Forest
plt.scatter(y_test, y_pred_rfr, alpha=0.6, color='green', label='Random Forest')

# Reference Line: Perfect prediction (45° line)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='-', linewidth=2, label='Perfect Prediction (45°)')

# Trendline: Linear Regression
lr_trend = np.poly1d(np.polyfit(y_test, y_pred_lr, deg=1))
plt.plot(y_test, lr_trend(y_test), color='blue', linestyle='--', linewidth=2, label='LR Trendline')

# Trendline: Random Forest
rfr_trend = np.poly1d(np.polyfit(y_test, y_pred_rfr, deg=1))
plt.plot(y_test, rfr_trend(y_test), color='green', linestyle='--', linewidth=2, label='RFR Trendline')

# Final plot settings
plt.title(" Actual vs Predicted Prices: Linear vs Random Forest", fontsize=14)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()