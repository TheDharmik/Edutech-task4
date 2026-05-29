# ============================================================
#  Task 4: Linear Regression – House Price Prediction
#  Edutech Solution – Data Science Internship
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("   House Price Prediction – Linear Regression")
print("=" * 55)

# ── STEP 2: Load Dataset ──────────────────────────────────────
df = pd.read_csv("HousePrice.csv")

print(f"\n📦 Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\n🔍 First 5 rows:")
print(df.head())

print("\n📊 Basic Statistics:")
print(df.describe().round(2))

print("\n✅ Missing values:", df.isnull().sum().sum(), "(none – good!)")

# ── STEP 3: Define Features (X) and Target (y) ───────────────
#   Features  → what the model uses to make predictions
#   Target    → what we want to predict (Price)

X = df.drop(columns=["Price"])   # all columns except Price
y = df["Price"]                   # only Price

print(f"\n🎯 Features used : {list(X.columns)}")
print(f"🎯 Target column : Price")

# ── STEP 4: Split into Training & Testing Sets ────────────────
#   80% data → model learns from this (training)
#   20% data → we test how well it learned (testing)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✂️  Training samples : {len(X_train)}")
print(f"✂️  Testing  samples : {len(X_test)}")

# ── STEP 5: Train the Linear Regression Model ─────────────────
model = LinearRegression()
model.fit(X_train, y_train)   # ← the model LEARNS here

print("\n✅ Model trained successfully!")

# ── STEP 6: Predict on the Test Set ──────────────────────────
y_pred = model.predict(X_test)

# ── STEP 7: Evaluate – RMSE & R² ─────────────────────────────
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 55)
print("📈  MODEL PERFORMANCE")
print("=" * 55)
print(f"  RMSE (Root Mean Square Error) : ₹{rmse:,.0f}")
print(f"  R² Score (Accuracy)           : {r2:.4f}  ({r2*100:.2f}%)")
print("=" * 55)
print("\n💡 What this means:")
print(f"   • On average, predictions are off by ₹{rmse:,.0f}")
print(f"   • The model explains {r2*100:.1f}% of price variation")

# ── STEP 8: Show Feature Importance (Coefficients) ───────────
coeff_df = pd.DataFrame({
    "Feature"    : X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("\n📌 Feature Coefficients (impact on price):")
print(coeff_df.to_string(index=False))
print(f"\n📌 Intercept (base price): ₹{model.intercept_:,.0f}")

# ── STEP 9: Predicted Values Report ──────────────────────────
report = X_test.copy()
report["Actual_Price"]    = y_test.values
report["Predicted_Price"] = y_pred.round(0)
report["Error"]           = (report["Predicted_Price"] - report["Actual_Price"]).round(0)
report["Error_%"]         = ((report["Error"] / report["Actual_Price"]) * 100).round(2)

report.to_csv("predicted_values_report.csv", index=False)
print("\n📄 Predicted values report saved → predicted_values_report.csv")
print(report.head(10).to_string(index=False))

# ── STEP 10: Visualisations ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("House Price – Linear Regression Results", fontsize=16, fontweight="bold")

# Plot 1: Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.5, color="steelblue", edgecolors="white", linewidth=0.5)
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Prices")
ax.legend()

# Plot 2: Residuals
residuals = y_test.values - y_pred
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.5, color="coral")
ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
ax.set_xlabel("Predicted Price")
ax.set_ylabel("Residual (Actual − Predicted)")
ax.set_title("Residual Plot")

# Plot 3: Feature Coefficients
ax = axes[1, 0]
colors = ["green" if c > 0 else "red" for c in coeff_df["Coefficient"]]
ax.barh(coeff_df["Feature"], coeff_df["Coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient Value")
ax.set_title("Feature Importance (Coefficients)")

# Plot 4: Price Distribution
ax = axes[1, 1]
ax.hist(y_test,  bins=30, alpha=0.6, label="Actual",    color="steelblue")
ax.hist(y_pred,  bins=30, alpha=0.6, label="Predicted", color="orange")
ax.set_xlabel("Price")
ax.set_ylabel("Count")
ax.set_title("Actual vs Predicted Price Distribution")
ax.legend()

plt.tight_layout()
plt.savefig("linear_regression_results.png", dpi=150, bbox_inches="tight")
print("\n📊 Visualisation saved → linear_regression_results.png")

print("\n🎉 Task Complete! All deliverables are ready.")
print("   ✅ Trained model   (in memory / script)")
print("   ✅ predicted_values_report.csv")
print("   ✅ linear_regression_results.png")
