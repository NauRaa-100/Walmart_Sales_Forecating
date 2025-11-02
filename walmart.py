"""

-- Walmart Sales Forecast dataset

"""

#Importing Libraries --
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression , Lasso ,Ridge , ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor
from xgboost import XGBRegressor
from prophet import Prophet
import statsmodels.api as sm
import joblib
import warnings
warnings.filterwarnings("ignore")




#Load Data --
df_features=pd.read_csv(r'E:\Rev-DataScience\AI-ML\MLProjects_Structured_Data\features.csv',encoding='latin')
df_stores=pd.read_csv(r'E:\Rev-DataScience\AI-ML\MLProjects_Structured_Data\stores.csv',encoding='latin')
df_train=pd.read_csv(r'E:\Rev-DataScience\AI-ML\MLProjects_Structured_Data\train.csv',encoding='latin')
df_test=pd.read_csv(r'E:\Rev-DataScience\AI-ML\MLProjects_Structured_Data\test.csv',encoding='latin')

#====================================================
# Merging of dataframes - creating using data --
#====================================================


# Merge
df_ = df_train.merge(df_features, on=["Store", "Date", "IsHoliday"], how="left")
df_ = df_.merge(df_stores, on="Store", how="left")
#---
#test = df_test.merge(df_features, on=["Store", "Date", "IsHoliday"], how="left")
#test = test.merge(df_stores, on="Store", how="left")

print(df_.head(5))
print(df_.dtypes)
print(df_.shape)
print(df_.isnull().sum())
print(df_.describe())



#======================================
# Cleaning Data
#======================================

#------------Optimization & Missing Values--------------

print(df_.memory_usage(deep=True))

for col in df_.columns:
    if df_[col].dtype == 'object':
        df_[col]=df_[col].astype('category')


for col in df_.columns:
    if df_[col].dtype=='float64':
        df_[col]=df_[col].astype('float32')


for col in df_.columns:
    if df_[col].dtype == 'int64':
        df_[col]=df_[col].astype('int8') 

nums=[]

for col in df_.columns:
    if df_[col].dtype in ['int8','int64','float32']:
       nums.append(col)

print(nums)
print(df_.memory_usage(deep=True))
#Check Null Values
print(df_.isnull().sum())

print('---------Seperate--------')

#==========================================
# Outliers --
#==========================================

def find_outliers(series):

    Q1= series.quantile(0.25)
    Q3=series.quantile(0.75)

    IQR= Q3 - Q1
    lower= Q1 - 1.5 * IQR
    upper= Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)]

for col in df_.columns:
    if df_[col].dtype in ['int8','float32']:
      print(f"Outliers in {col} => {find_outliers(df_[col]).shape[0]}")
      print('----------------------------')
      #------Visualization of outliers-------

      plt.boxplot(df_[col])
      plt.title(f"Outliers in {col}")
      plt.show()
      



"""

Overview Insights --
-- Shape of dataframe  =>
rows = 421570 , columns= 7

-- Empty Values --
There are no null values
-- there are many outliers in target column and will keep it 

*-* Actions
-- optimize data 


"""

df=df_.copy()
#Editing datetime column

df['Date']=pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year
df['Week']=df['Date'].dt.day_of_week

#----

print(df['Weekly_Sales'].value_counts())

print(df['Year'].value_counts())
# 2010 , 2011 , 2012 the most repeted

print(df['Week'].value_counts())
# Week 4 most repeted


#Simple Analysis

for col in df.columns:
   if col != 'Weekly_Sales':
     print(f"Relation between {col} and Weekly sales =:\n{df.groupby(col)['Weekly_Sales'].median()}")
     print('-----------------------------')
      
     plt.figure(figsize=(9,6))
     sns.barplot(x=col,y='Weekly_Sales',data=df)
     plt.title(f"Relation between {col} and Weekly Sales")
     plt.show()
    
#===============================================
# EDA (Exploratory Data Analysis)
#===============================================

plt.figure(figsize=(8,5))
sns.histplot(df['Weekly_Sales'], kde=True)
plt.title('Weekly Sales Distribution (Before Transform)')
plt.show()

print("Before log => ",df['Weekly_Sales'].skew())

if df['Weekly_Sales'].skew() > 0.5:
    df['Weekly_Sales']=np.log1p(df['Weekly_Sales'])

plt.figure(figsize=(8,5))
sns.histplot(df['Weekly_Sales'],kde=True,color='red')
plt.title('Distribution (After Transform)')
plt.show()



#========================================
# Encoding --
#========================================

df_enc=pd.get_dummies(df,drop_first=True)
df_enc.dropna(subset='Weekly_Sales')

#----

#Check Null Values
print(df_enc.isnull().sum())

#========================================
# Scaling 
#========================================


x=df_enc.drop('Weekly_Sales',axis=1).values
y=df_enc['Weekly_Sales'].values

scaler=RobustScaler()
x_scaled= scaler.fit_transform(x)

#------Split--------
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

#========================================
# Features Selection --
#========================================

#Indexing 
df_t=df_enc.drop('Weekly_Sales',axis=1)

select=RandomForestRegressor(random_state=42,n_estimators=100,n_jobs=-1)
select.fit(x_train,y_train)

importances=pd.Series(select.feature_importances_,index=df_t.columns).sort_values(ascending=False)

print(importances.head(5))

#----Visualization of top features----


plt.figure(figsize=(12,6))
sns.barplot(x=importances.values[:5],y=importances.index[:5])
plt.title("TOP Features")
plt.show()

feats=['Dept','Store','Type_B','Size','Type_C']


#============================================
#  ML Models
#============================================

x=df_enc[feats].values
y=df_enc['Weekly_Sales'].values

x_scaled=RobustScaler().fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,shuffle=False,test_size=0.2,random_state=42)


ml_models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(random_state=42),
    'Ridge': Ridge(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(),
    'Gradient':GradientBoostingRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}

print("\n==================== Structured ML Models ====================")
for name, model in ml_models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"{name}: R2={r2_score(y_test, y_pred):.3f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")


#============================================
#  Time Series Models
#============================================

print("\n==================== Time-Series Models ====================")

ts_df = df_[['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
train_size = int(len(ts_df) * 0.8)
train, test = ts_df.iloc[:train_size], ts_df.iloc[train_size:]
df_ = df_.sort_values('Date')
train['y'] = train['y'].astype(float)


# ARIMA
arima = ARIMA(train['y'], order=(1,1,1))
arima_fit = arima.fit()
arima_pred = arima_fit.forecast(steps=len(test))
print("ARIMA: R2 =", r2_score(test['y'], arima_pred))

# SARIMA
sarima = SARIMAX(train['y'], order=(1,1,1), seasonal_order=(1,1,1,52))
sarima_fit = sarima.fit()
sarima_pred = sarima_fit.forecast(steps=len(test))
print("SARIMA: R2 =", r2_score(test['y'], sarima_pred))



#The Best Model is GradientBoosting 

#Applying Best Model 

model= GradientBoostingRegressor(random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
acc=model.score(x_test,y_test)
print("Accuracy => ",acc)


plt.scatter(y_test,y_pred,c=y_pred,cmap='coolwarm')
plt.xlabel('Actual')
plt.ylabel("Predicted")
plt.show()


residuals=y_test - y_pred 
sns.histplot(residuals,color='red')
plt.title("Reisuals")
plt.show()


#==============================================
# Saving
#==============================================

joblib.dump(model, 'walmart_best_model.pkl')
joblib.dump(RobustScaler(), 'scaler.pkl')
print("Saved")

