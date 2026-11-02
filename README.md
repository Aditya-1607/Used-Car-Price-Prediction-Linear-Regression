# 🚗 Used Car Price Prediction with Linear Regression

A beginner-friendly machine learning project that predicts **used Toyota Corolla prices** using vehicle details and **Linear Regression**, built with pandas and scikit-learn.

---

## 📌 Overview

This project demonstrates an end-to-end **supervised regression** pipeline using the Toyota Corolla dataset stored locally as `ToyotaCorolla.csv`.

The project covers:

1. Loading the dataset from a CSV file
2. Exploring the dataset using summary statistics
3. Selecting relevant numerical features
4. Separating input features (`X`) from the target (`y`)
5. Splitting the data into training and testing sets
6. Training a Linear Regression model
7. Evaluating the model using MSE and R² score
8. Viewing the model's intercept and coefficients
9. Predicting the price for a custom car input

---

## 🎯 Problem Statement

**Goal:** Predict the selling price of a used Toyota Corolla based on its vehicle age, mileage, horsepower, engine capacity, number of doors, and weight.

- **Target variable:** `Price` — used car selling price.
- **Input features:** `Age`, `KM`, `HP`, `CC`, `Doors`, and `Weight`.

The model uses these six numerical features to predict the car's price.

---

## 📂 Dataset

The project uses the local file **`ToyotaCorolla.csv`** instead of loading the dataset from an online URL.

```python
df = pd.read_csv('ToyotaCorolla.csv')
```

- **Dataset:** Toyota Corolla Used Car Dataset
- **Rows:** 1,436
- **Columns:** 10
- **Target column:** `Price`

### Dataset Columns

| Column | Description |
|--------|-------------|
| `Price` | Selling price of the car (target variable) |
| `Age` | Age of the car in months |
| `KM` | Total distance driven in kilometres |
| `FuelType` | Type of fuel used |
| `HP` | Horsepower |
| `MetColor` | Metallic colour indicator |
| `Automatic` | Automatic transmission indicator |
| `CC` | Engine capacity in cc |
| `Doors` | Number of doors |
| `Weight` | Weight of the car in kg |

### Features Used in the Model

The model selects the following numerical columns:

`Age`, `KM`, `HP`, `CC`, `Doors`, and `Weight`.

`FuelType`, `MetColor`, and `Automatic` are present in the dataset but are not used in this model.

---

## 🛠️ Requirements

- Python 3.8 or higher
- pandas
- scikit-learn
- Jupyter Notebook (optional)

Install the required libraries:

```bash
pip install pandas scikit-learn jupyter
```

---

## 🚀 Usage

Keep `ToyotaCorolla.csv` in the same folder as your Jupyter Notebook.

### 1. Import libraries and load the dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('ToyotaCorolla.csv')

features = ['Age', 'KM', 'HP', 'CC', 'Doors', 'Weight']
df = df[['Price'] + features]

df.head()
```

### 2. Explore the dataset

```python
df.describe()
```

### 3. Separate features and target

```python
X = df.drop('Price', axis=1)
y = df['Price']
```

- `X` contains the six input features.
- `y` contains the car prices that the model must predict.

### 4. Split into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **80%** of the data is used for training.
- **20%** of the data is used for testing.
- `random_state=42` makes the split reproducible.

### 5. Train the Linear Regression model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### 6. Make predictions and evaluate the model

```python
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
```

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted prices. Lower is better.
- **R² Score:** Measures how well the model explains variation in car prices. A value closer to 1 indicates a better fit.

### 7. View the learned model coefficients

```python
print("Intercept:", model.intercept_)

weights = pd.DataFrame(
    model.coef_, X.columns, columns=['Weight']
)
weights
```

The intercept and coefficients form the equation used by the Linear Regression model to make predictions.

### 8. Predict the price for a custom car

```python
new_sample = pd.DataFrame({
    'Age': [45],
    'KM': [60000],
    'HP': [110],
    'CC': [1600],
    'Doors': [5],
    'Weight': [1100]
})

predicted_price = model.predict(new_sample)

print("Predicted Price:", predicted_price[0])
```

The custom input must contain the same feature names and order used during model training.

---

## 📊 Results

The notebook prints the following evaluation metrics after testing:

| Metric | Meaning |
|--------|---------|
| **Mean Squared Error (MSE)** | Average squared prediction error |
| **R² Score** | Proportion of variation in car prices explained by the model |

The actual metric values depend on the dataset and the train-test split. Run the notebook to obtain the results; no fixed accuracy value is assumed here.

---

## 🧠 Model Interpretation

The model coefficients indicate how each input feature contributes to the predicted price, while keeping the other features constant.

The coefficient table is generated by:

```python
weights = pd.DataFrame(
    model.coef_, X.columns, columns=['Weight']
)
```

- A positive coefficient indicates that an increase in that feature increases the model's predicted price, all else being equal.
- A negative coefficient indicates that an increase in that feature decreases the model's predicted price, all else being equal.
- The magnitude of a coefficient depends on the feature's scale and units.

---

## 📁 Project Structure

```text
├── ToyotaCorolla.csv    # Dataset
├── notebook1.ipynb      # Jupyter Notebook containing the ML pipeline
└── README.md            # Project documentation
```

---

## 💡 Key Takeaways

- Linear Regression can be used to predict used car prices.
- The dataset is loaded from a local CSV file.
- Six numerical vehicle features are used for prediction.
- The model is evaluated using MSE and R² score.
- A custom car record can be used to generate a price prediction.

---

## 🙏 Acknowledgements

- [pandas](https://pandas.pydata.org/) — data loading and manipulation
- [scikit-learn](https://scikit-learn.org/) — model training and evaluation
