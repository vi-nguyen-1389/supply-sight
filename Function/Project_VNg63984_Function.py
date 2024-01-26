
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
import os
import json
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly as py
import pandas as pd


# In[1]:

# function to manually select the features
def select_manual(data, target, threshold):
    corr_target = data.corr().abs()
    selected_features = corr_target[corr_target[target]
                                    >= threshold].index.tolist()
    selected_features.remove(target)
    return data[selected_features]


# In[2]:


# function to select features using the variance threshold method
def select_variance(data, target, threshold):
    var_threshold = VarianceThreshold(threshold)
    var_threshold.fit(data)
    selected_features = data.iloc[:,
                                  var_threshold.get_support()].columns.tolist()
    if target in selected_features:
        selected_features.remove(target)
    return data[selected_features]


# In[3]:
def select_best(X_train, y_train, num):
    kbest = SelectKBest(score_func=f_regression, k=num)
    kbest.fit_transform(X_train, y_train)
    selected_columns = X_train.columns[kbest.get_support()]
    return selected_columns

# In[4]:


def make_poly(data):
    poly = PolynomialFeatures(
        degree=2, include_bias=False, interaction_only=True)
    return poly.fit_transform(data)


# function to perform linear regression
def perform_linear_regression(features, target, test_size, random_state, transformation=None):
    scaler = StandardScaler()
    model = LinearRegression()

    if transformation == 'Poly 2 Interaction':
        features = make_poly(features)

    if transformation == 'PCA':
        pca = PCA(n_components=0.95)  # retain 95% of the variance
        features = pca.fit_transform(features)

    X_scaled = scaler.fit_transform(features)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, target, test_size=test_size, random_state=random_state)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    r2 = model.score(X_test, Y_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)

    return r2, rmse, model, X_test, Y_test

# function for ADA regression


def perform_ada_regression(features, target, test_size, random_state, transformation=None):
    scaler = StandardScaler()
    model = AdaBoostRegressor()

    if transformation == 'Poly 2 Interaction':
        features = make_poly(features)

    if transformation == 'PCA':
        pca = PCA(n_components=0.95)  # retain 95% of the variance
        features = pca.fit_transform(features)

    X_scaled = scaler.fit_transform(features)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, target, test_size=test_size, random_state=random_state)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    r2 = model.score(X_test, Y_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)

    return r2, rmse, model, X_test, Y_test

# function for random forest regression


def perform_rf_regression(features, target, test_size, random_state, transformation=None):
    scaler = StandardScaler()
    model = RandomForestRegressor()

    if transformation == 'Poly 2 Interaction':
        features = make_poly(features)

    if transformation == 'PCA':
        pca = PCA(n_components=0.95)  # retain 95% of the variance
        features = pca.fit_transform(features)

    X_scaled = scaler.fit_transform(features)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, target, test_size=test_size, random_state=random_state)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    r2 = model.score(X_test, Y_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)

    return r2, rmse, model, X_test, Y_test

# function for XGB regression


def perform_xgb_regression(features, target, test_size, random_state, transformation=None):
    scaler = StandardScaler()
    model = XGBRegressor()

    if transformation == 'Poly 2 Interaction':
        features = make_poly(features)

    if transformation == 'PCA':
        pca = PCA(n_components=0.95)  # retain 95% of the variance
        features = pca.fit_transform(features)

    X_scaled = scaler.fit_transform(features)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, target, test_size=test_size, random_state=random_state)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    r2 = model.score(X_test, Y_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)

    return r2, rmse, model, X_test, Y_test

# function for decision tree regression


def perform_dt_regression(features, target, test_size, random_state, transformation=None):
    scaler = StandardScaler()
    model = DecisionTreeRegressor()

    if transformation == 'Poly 2 Interaction':
        features = make_poly(features)

    if transformation == 'PCA':
        pca = PCA(n_components=0.95)  # retain 95% of the variance
        features = pca.fit_transform(features)

    X_scaled = scaler.fit_transform(features)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, target, test_size=test_size, random_state=random_state)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    r2 = model.score(X_test, Y_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)

    return r2, rmse, model, X_test, Y_test


# function for Ridge Regression

def perform_ridge_regression(X_train, X_test, Y_train, Y_test, alpha):
    ridge = Ridge(alpha=alpha, max_iter=1000)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)
    r2 = ridge.score(X_test, Y_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    return r2, rmse, ridge, X_test, Y_test


# function for Lasso Regression
def perform_lasso_regression(X_train, X_test, Y_train, Y_test, alpha):
    lasso = Lasso(alpha=alpha, max_iter=1000)
    lasso.fit(X_train, Y_train)
    Y_pred = lasso.predict(X_test)
    r2 = lasso.score(X_test, Y_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False) ** 0.5
    return r2, rmse, lasso, X_test, Y_test


def proper_case_linear_algorithm_name(name):
    name_mapping = {
        "adaboost": "AdaBoost",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "decision_tree": "Decision Tree",
        "lasso": "Lasso",
        "ridge": "Ridge",
        "linear": "Linear Regression"
    }
    return name_mapping.get(name, name)
