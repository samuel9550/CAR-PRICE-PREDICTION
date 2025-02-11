import pandas as pd

# Load dataset
data = pd.read_csv("Car details v3.csv")

# ✅ Drop unnecessary columns
data.drop(columns=["name", "torque"], inplace=True, errors='ignore')
print("\n✅ Dropped 'name' and 'torque' columns.")

# ✅ Convert 'mileage', 'engine', and 'max_power' to numeric values
data["mileage"] = pd.to_numeric(data["mileage"].str.extract(r'([\d\.]+)')[0], errors="coerce")
data["engine"] = pd.to_numeric(data["engine"].str.extract(r'([\d\.]+)')[0], errors="coerce")
data["max_power"] = pd.to_numeric(data["max_power"].str.extract(r'([\d\.]+)')[0], errors="coerce")
print("\n✅ Converted mileage, engine, and max_power to numeric.")

# ✅ Fill missing values with median
data.fillna(data.median(numeric_only=True), inplace=True)
print("\n✅ Filled missing values with median.")

# ✅ Convert 'seats' column to integer
data["seats"] = data["seats"].astype(int, errors='ignore')
print("\n✅ Converted 'seats' column to integer.")

# ✅ One-hot encode categorical variables
data = pd.get_dummies(data, columns=["fuel", "seller_type", "transmission", "owner"], drop_first=True)
print("\n✅ Applied one-hot encoding to categorical variables.")

# ✅ Display processed dataset info
print("\n🔹 Processed Dataset Info:")
print(data.info())

# ✅ Show first few rows of the processed dataset
print("\n🔹 Processed Dataset Preview:")
print(data.head())
import pandas as pd

# Load dataset
data = pd.read_csv("Car details v3.csv")

# ✅ Drop unnecessary columns
data.drop(columns=["name", "torque"], inplace=True, errors="ignore")
print("\n✅ Dropped 'name' and 'torque' columns.")

# ✅ Convert 'mileage', 'engine', and 'max_power' to numeric values
data["mileage"] = pd.to_numeric(data["mileage"].str.extract(r'([\d\.]+)')[0], errors="coerce")
data["engine"] = pd.to_numeric(data["engine"].str.extract(r'([\d\.]+)')[0], errors="coerce")
data["max_power"] = pd.to_numeric(data["max_power"].str.extract(r'([\d\.]+)')[0], errors="coerce")
print("\n✅ Converted mileage, engine, and max_power to numeric.")

# ✅ Fill missing values with median
data.fillna(data.median(numeric_only=True), inplace=True)
print("\n✅ Filled missing values with median.")

# ✅ Convert 'seats' column to integer
if "seats" in data.columns:
    data["seats"] = data["seats"].astype(int, errors="ignore")
print("\n✅ Converted 'seats' column to integer.")

# ✅ One-hot encode categorical variables
data = pd.get_dummies(data, columns=["fuel", "seller_type", "transmission", "owner"], drop_first=True)
print("\n✅ Applied one-hot encoding to categorical variables.")

# ✅ Display processed dataset info
print("\n🔹 Processed Dataset Info:")
print(data.info())

# ✅ Show first few rows of the processed dataset
print("\n🔹 Processed Dataset Preview:")
print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# ✅ Split dataset into features (X) and target variable (y)
X = data.drop(columns=["selling_price"])  # All columns except target
y = data["selling_price"]  # Target variable

# ✅ Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✅ Split data into training and testing sets.")

# ✅ Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✅ Model training complete.")

# ✅ Save the trained model using Pickle
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\n✅ Model saved as 'car_price_model.pkl'")
from sklearn.metrics import r2_score, mean_absolute_error

# ✅ Make Predictions
y_pred = model.predict(X_test)
print("\n✅ Predictions made on the test set.")

# ✅ Evaluate Model Performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n🔹 Model Evaluation Metrics:")
print(f"✅ R² Score: {r2:.4f} (higher is better)")
print(f"✅ Mean Absolute Error (MAE): ₹{mae:.2f} (lower is better)")
