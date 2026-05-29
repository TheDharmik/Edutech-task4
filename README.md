**Edutech Solution | Data Science Internship | Task 4**

---

## 📌 Objective

Build a Linear Regression model to predict house prices based on features like area, number of bedrooms, bathrooms, and distance from the city.

---

## 📁 Repository Structure

```
├── house_price_linear_regression.py   # Main Python script
├── HousePrice.csv                     # Dataset (500 records)
├── predicted_values_report.csv        # Model predictions vs actual prices
├── linear_regression_results.png      # Visualisation charts
└── README.md                          # This file
```

---

## 🗂️ Dataset

A synthetic House Price dataset with **500 rows** and the following columns:

| Column | Description |
|---|---|
| `Area_sqft` | Size of the house in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Age_years` | Age of the house in years |
| `Garage` | Number of garage spaces |
| `Distance_city_km` | Distance from city centre (km) |
| `Price` | House price in ₹ *(target variable)* |

---

## 🛠️ Tools & Libraries

- **Python 3**
- **scikit-learn** — model training & evaluation
- **pandas** — data handling
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualisations

---

## ⚙️ How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```

3. Run the script
   ```bash
   python house_price_linear_regression.py
   ```

---

## 🔄 Steps Followed

1. Loaded and explored the dataset
2. Defined features (X) and target variable (y = Price)
3. Split data → **80% training / 20% testing**
4. Trained a `LinearRegression` model using scikit-learn
5. Predicted house prices on the test set
6. Evaluated performance using **RMSE** and **R² Score**
7. Saved predictions to `predicted_values_report.csv`
8. Plotted results and saved to `linear_regression_results.png`

---

## 📈 Results

| Metric | Value |
|---|---|
| **RMSE** (Root Mean Square Error) | ₹32,069 |
| **R² Score** | 0.9375 (93.75%) |

The model explains **93.75%** of the variation in house prices, with an average prediction error of ₹32,069.

---





## 🙋 Interview Questions

**Q: What is regression?**
Regression is a supervised machine learning technique used to predict a continuous numerical output based on one or more input features. Linear regression models the relationship as a straight line.

**Q: What is RMSE and why is it used?**
RMSE (Root Mean Square Error) is the square root of the average of squared differences between predicted and actual values. It is used because it is in the same unit as the target variable, making it easy to interpret, and it penalises large errors more than small ones.

