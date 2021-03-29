import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import math
import statsmodels.api as sm
from scipy import stats
import regressors
from sklearn.metrics import mean_squared_log_error
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = None #para retirar limite de colunas do pandas
pd.options.display.max_rows = None #para retirar limite de linhas do pandas

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.info())
print(test.info())

print(train.describe())
print(test.describe())

# Verificando missing numbers
nanTrain = train.isnull().sum().sort_values(ascending=False)
nanTrain.head(20)
nanTest = test.isnull().sum().sort_values(ascending=False)
nanTest.head(35)

# Para colunas com mais de 50% de valores nulos serão descondideradas no modelo
train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)

# Para os demais será focado na média e no valor mais comum

# Por média
train['LotFrontage'] = train['LotFrontage'].fillna((train['LotFrontage'].mean()))
test['LotFrontage'] = test['LotFrontage'].fillna((test['LotFrontage'].mean()))

# Por valor mais comum
train['GarageYrBlt'].value_counts().idxmax()
train['GarageYrBlt'] = train['GarageYrBlt'].fillna((train['GarageYrBlt'].value_counts().idxmax()))
test['GarageYrBlt'].value_counts().idxmax()
test['GarageYrBlt'] = test['GarageYrBlt'].fillna((test['GarageYrBlt'].value_counts().idxmax()))

train['GarageType'].value_counts().idxmax()
train['GarageType'] = train['GarageType'].fillna((train['GarageType'].value_counts().idxmax()))
test['GarageType'].value_counts().idxmax()
test['GarageType'] = test['GarageType'].fillna((test['GarageType'].value_counts().idxmax()))

train['GarageQual'].value_counts().idxmax()
train['GarageQual'] = train['GarageQual'].fillna((train['GarageQual'].value_counts().idxmax()))
test['GarageQual'].value_counts().idxmax()
test['GarageQual'] = test['GarageQual'].fillna((test['GarageQual'].value_counts().idxmax()))

train['GarageCond'].value_counts().idxmax()
train['GarageCond'] = train['GarageCond'].fillna((train['GarageCond'].value_counts().idxmax()))
test['GarageCond'].value_counts().idxmax()
test['GarageCond'] = test['GarageCond'].fillna((test['GarageCond'].value_counts().idxmax()))

train['GarageFinish'].value_counts().idxmax()
train['GarageFinish'] = train['GarageFinish'].fillna((train['GarageFinish'].value_counts().idxmax()))
test['GarageFinish'].value_counts().idxmax()
test['GarageFinish'] = test['GarageFinish'].fillna((test['GarageFinish'].value_counts().idxmax()))

test['GarageFinish'] = test['GarageFinish'].fillna((test['GarageFinish'].value_counts().idxmax()))
train['BsmtCond'] = train['BsmtCond'].fillna((train['BsmtCond'].value_counts().idxmax()))
test['BsmtCond'].value_counts().idxmax()
test['BsmtCond'] = test['BsmtCond'].fillna((test['BsmtCond'].value_counts().idxmax()))

train['BsmtFinType2'].value_counts().idxmax()
train['BsmtFinType2'] = train['BsmtFinType2'].fillna((train['BsmtFinType2'].value_counts().idxmax()))
test['BsmtFinType2'].value_counts().idxmax()
test['BsmtFinType2'] = test['BsmtFinType2'].fillna((test['BsmtFinType2'].value_counts().idxmax()))

train['BsmtExposure'].value_counts().idxmax()
train['BsmtExposure'] = train['BsmtExposure'].fillna((train['BsmtExposure'].value_counts().idxmax()))
test['BsmtExposure'].value_counts().idxmax()
test['BsmtExposure'] = test['BsmtExposure'].fillna((test['BsmtExposure'].value_counts().idxmax()))

train['BsmtFinType1'].value_counts().idxmax()
train['BsmtFinType1'] = train['BsmtFinType1'].fillna((train['BsmtFinType1'].value_counts().idxmax()))
test['BsmtFinType1'].value_counts().idxmax()
test['BsmtFinType1'] = test['BsmtFinType1'].fillna((test['BsmtFinType1'].value_counts().idxmax()))

train['BsmtQual'].value_counts().idxmax()
train['BsmtQual'] = train['BsmtQual'].fillna((train['BsmtQual'].value_counts().idxmax()))
test['BsmtQual'].value_counts().idxmax()
test['BsmtQual'] = test['BsmtQual'].fillna((test['BsmtQual'].value_counts().idxmax()))

train['MasVnrType'].value_counts().idxmax()
train['MasVnrType'] = train['MasVnrType'].fillna((train['MasVnrType'].value_counts().idxmax()))
test['MasVnrType'].value_counts().idxmax()
test['MasVnrType'] = test['MasVnrType'].fillna((test['MasVnrType'].value_counts().idxmax()))

train['MasVnrArea'].value_counts().idxmax()
train['MasVnrArea'] = train['MasVnrArea'].fillna((train['MasVnrArea'].value_counts().idxmax()))
test['MasVnrArea'].value_counts().idxmax()
test['MasVnrArea'] = test['MasVnrArea'].fillna((test['MasVnrArea'].value_counts().idxmax()))

# Removendo valores nan unicos de cada dataset
train['Electrical'].value_counts().idxmax()
train['Electrical'] = train['Electrical'].fillna((train['Electrical'].value_counts().idxmax()))

test['MSZoning'].value_counts().idxmax()
test['MSZoning'] = train['MSZoning'].fillna((train['MSZoning'].value_counts().idxmax()))

test['BsmtFullBath'].value_counts().idxmax()
test['BsmtFullBath'] = train['BsmtFullBath'].fillna((train['BsmtFullBath'].value_counts().idxmax()))

test['BsmtHalfBath'].value_counts().idxmax()
test['BsmtHalfBath'] = train['BsmtHalfBath'].fillna((train['BsmtHalfBath'].value_counts().idxmax()))

test['Functional'].value_counts().idxmax()
test['Functional'] = train['Functional'].fillna((train['Functional'].value_counts().idxmax()))

test['Utilities'].value_counts().idxmax()
test['Utilities'] = train['Utilities'].fillna((train['Utilities'].value_counts().idxmax()))

test['TotalBsmtSF'].value_counts().idxmax()
test['TotalBsmtSF'] = train['TotalBsmtSF'].fillna((train['TotalBsmtSF'].value_counts().idxmax()))

test['BsmtFinSF2'].value_counts().idxmax()
test['BsmtFinSF2'] = train['BsmtFinSF2'].fillna((train['BsmtFinSF2'].value_counts().idxmax()))

test['BsmtFinSF1'].value_counts().idxmax()
test['BsmtFinSF1'] = train['BsmtFinSF1'].fillna((train['BsmtFinSF1'].value_counts().idxmax()))

test['KitchenQual'].value_counts().idxmax()
test['KitchenQual'] = train['KitchenQual'].fillna((train['KitchenQual'].value_counts().idxmax()))

test['BsmtUnfSF'].value_counts().idxmax()
test['BsmtUnfSF'] = train['BsmtUnfSF'].fillna((train['BsmtUnfSF'].value_counts().idxmax()))

test['SaleType'].value_counts().idxmax()
test['SaleType'] = train['SaleType'].fillna((train['SaleType'].value_counts().idxmax()))

test['GarageCars'].value_counts().idxmax()
test['GarageCars'] = train['GarageCars'].fillna((train['GarageCars'].value_counts().idxmax()))

test['Exterior1st'].value_counts().idxmax()
test['Exterior1st'] = train['Exterior1st'].fillna((train['Exterior1st'].value_counts().idxmax()))

test['GarageArea'].value_counts().idxmax()
test['GarageArea'] = train['GarageArea'].fillna((train['GarageArea'].value_counts().idxmax()))

test['Exterior2nd'].value_counts().idxmax()
test['Exterior2nd'] = train['Exterior2nd'].fillna((train['Exterior2nd'].value_counts().idxmax()))

# Não restou nenhum missing number

# Verificando as correlações das variáveis
plt.figure(figsize=(16,6))
heatmap = sns.heatmap(train.corr(), vmin=-1, vmax=1, annot=False, cmap='BrBG')

# Removendo as correlações inferiores a 0,6 em módulo
train.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea',
           '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
           'MoSold', 'YrSold'], axis=1, inplace=True)
test.drop(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea',
           '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
           'MoSold', 'YrSold'], axis=1, inplace=True)

# Verificando como ficou após a exclusão das correlações e multicorrelações
plt.figure(figsize=(16,6))
heatmap = sns.heatmap(train.corr(), vmin=-1, vmax=1, annot=False, cmap='BrBG')

# Com a retirada das variáveis não correlacionadas não restou variáveis multicolineriaezadas
X = train[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars']]
y = train['SalePrice']
model = sm.OLS.from_formula("y ~ X", data=train)
result  = model.fit()
result.summary(xname=['interceptador', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars'])

reg = LinearRegression().fit(X, y)
X_test = test[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars']]
pred = reg.predict(X_test)

pred_train = reg.predict(X)

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

print('mape: ', mape(y, pred_train))

# Analisando outliers
X = train[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars']]

fig = go.Figure()

for col in X:
  fig.add_trace(go.Box(y=train[col].values, name=train[col].name))
  
fig.show()

# Existem outliers em todas as variáveis
train = train[train.GrLivArea < 2630]
train = train[train.TotalBsmtSF < 1935]
train = train[train['1stFlrSF'] < 2117]

# Outliers removidos ou reduzidos
#rabalhando com variáveis categóricas¶

for item in train:
    if item == 'Id' or item == 'OverallQual' or item == 'TotalBsmtSF' or item == '1stFlrSF' or item == 'GrLivArea' or item == 'GarageCars' or item == 'SalePrice':
        pass
    else:
        px.box(train, x=item, y="SalePrice").show()

train.drop(['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
           'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
           'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterCond', 'Foundation', 'BsmtCond',
           'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical',
           'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
           'PavedDrive', 'SaleType', 'SaleCondition', 'Heating', 'BsmtExposure'], axis=1, inplace=True)

test.drop(['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
           'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
           'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterCond', 'Foundation', 'BsmtCond',
           'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical',
           'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
           'PavedDrive', 'SaleType', 'SaleCondition', 'Heating', 'BsmtExposure'], axis=1, inplace=True)

MSZoningTrain = pd.get_dummies(train['MSZoning'])
MSZoningTest = pd.get_dummies(test['MSZoning'])
MSZoningTrain.drop(['FV'], axis=1, inplace=True)
MSZoningTest.drop(['FV'], axis=1, inplace=True)

train = pd.concat([train, MSZoningTrain], axis=1)
test = pd.concat([test, MSZoningTest], axis=1)

train.drop(['MSZoning'], axis=1, inplace=True)
test.drop(['MSZoning'], axis=1, inplace=True)

# Testando o modelo com as novas variáveis¶
X = train[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'C (all)', 'RH', 'RL', 'RM']]
y = train['SalePrice']
model = sm.OLS.from_formula("y ~ X", data=train)
result  = model.fit()
result.summary(xname=['interceptador', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'C (all)', 'RH', 'RL', 'RM'])

reg = LinearRegression().fit(X, y)
X_test = test[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'C (all)', 'RH', 'RL', 'RM']]
pred = reg.predict(X_test)

pred_train = reg.predict(X)

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

print('mape: ', mape(y, pred_train))



