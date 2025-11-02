
#  Walmart Sales Forecasting

This project predicts **weekly sales** for Walmart stores using both **Machine Learning** and **Time Series** models.  
It includes **data preprocessing, feature selection, model comparison, and forecasting** using ARIMA, and SARIMA.

---

##  Dataset Overview

The dataset contains **store-level historical sales** data, with additional information about stores and features (holidays, CPI, temperature, etc.).

| File | Description |
|------|--------------|
| `features.csv` | External and seasonal factors (Temperature, CPI, Fuel Price, etc.) |
| `stores.csv` | Store details (Type, Size) |
| `train.csv` | Historical weekly sales for each store-department |
| `test.csv` | Test data for prediction submission |

---

##  Workflow

### ** Data Loading & Merging**
- Merged `train`, `features`, and `stores` on common keys.
- Combined into one clean dataframe `df_`.

### ** Data Cleaning & Optimization**
- Converted data types to reduce memory usage.
- Checked and handled missing values.
- Removed or visualized outliers using **IQR method**.

### ** Feature Engineering**
- Extracted `Year` and `Week` from the `Date` column.
- Encoded categorical features using `pd.get_dummies(drop_first=True)`.
- Scaled data using `RobustScaler()` to handle outliers.

### ** Exploratory Data Analysis (EDA)**
- Visualized relationships between features and sales.
- Checked the **distribution and skewness** of `Weekly_Sales`.
- Applied **log transformation** to normalize the target variable.

### **Feature Importance**
Used **RandomForestRegressor** to find the most important predictors.

Top features:
```

Dept
Store
Type_B
Size
Type_C

````
---

##  Time-Series Forecasting

Used `Date` and `Weekly_Sales` for time-based models.

- **ARIMA (1,1,1)**
- **SARIMA (1,1,1,52)** – accounting for weekly seasonality

Performance measured using R² on test set.

---

##  Best Model

 **Gradient Boosting Regressor** achieved the best accuracy.

```python
model = GradientBoostingRegressor(random_state=42)
model.fit(x_train, y_train)


Performance metrics:

```python
R2 = model.score(x_test, y_test)
RMSE = sqrt(mean_squared_error(y_test, y_pred))
```

---

##  Visualization

* **Outlier detection** (boxplots)
* **Feature importance** (barplot)
* **Actual vs Predicted** (scatter plot)
* **Residuals distribution**

Example:

```python
plt.scatter(y_test, y_pred, c=y_pred, cmap='coolwarm')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
```

---

##  Model Saving

Saved trained model and scaler for deployment:

```python
joblib.dump(model, "walmart_best_model.pkl")
joblib.dump(RobustScaler(), "scaler.pkl")
```

---

##  Tech Stack

| Category          | Tools / Libraries                         |
| ----------------- | ----------------------------------------- |
| **Languages**     | Python                                    |
| **Data Handling** | Pandas, NumPy                             |
| **Visualization** | Matplotlib, Seaborn                       |
| **Modeling**      | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| **Time Series**   | ARIMA, SARIMA, Prophet                    |
| **Model Saving**  | Joblib                                    |
| **Optimization**  | RobustScaler                              |

---

##  Key Insights

* Gradient Boosting provided the most stable and accurate predictions.
* Weekly seasonality plays a key role in sales behavior.
* Feature selection and robust scaling significantly improved results.
* ARIMA & SARIMA models capture the general trend but underperform compared to ML models.

---

##  Future Improvements

* Tune SARIMA hyperparameters more precisely.
* Add holiday and event indicators for Prophet model.
* Use SHAP or LIME for feature explainability.
* Deploy as a **Streamlit dashboard** for interactive forecasting.

---

##  Author

**Nau Raa**
Data Science & AI Learner
 *Project inspired by real-world Walmart sales data challenges.*

---

 *“Predicting sales isn’t just about numbers — it’s about understanding the rhythm of the market.”*
