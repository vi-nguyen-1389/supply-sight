# import necessary libraries
import smtplib
from flask import Flask, render_template, request, flash, redirect, url_for
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.decomposition import PCA
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from preprocessing import df_transformed, df_original, df_variance, df_selKBest, df, df_delaydetection, df1
from visualization import create_confusion_matrix_heatmap, create_visualizations

from Function.Project_VNg63984_Function import perform_linear_regression, perform_ada_regression, perform_rf_regression, perform_xgb_regression, perform_dt_regression, proper_case_linear_algorithm_name

import os

base_dir = os.path.abspath(os.path.dirname(__file__))
json_file_path = os.path.join(base_dir, 'analysis_texts.json')


application = Flask(__name__)


@application.route('/')
def main_index():
    # create_visualizations(df1)
    with open('analysis_texts.json', 'r') as file:
        analysis_texts = json.load(file)
    return render_template('index.html', analysis_texts=analysis_texts)


@application.route('/update-analysis', methods=['POST'])
def update_analysis():
    try:
        analysis_id = list(request.json.keys())[0]
        new_text = request.json[analysis_id]

        with open('analysis_texts.json', 'r') as file:
            analysis_texts = json.load(file)

        analysis_texts[analysis_id] = new_text

        with open('analysis_texts.json', 'w') as file:
            json.dump(analysis_texts, file)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@application.route('/sales-forecasting')
def sales_forecasting():
    return render_template('sales_forecasting.html')


@application.route('/delay-detection')
def delay_detection():
    return render_template('delay_detection.html')


@application.route('/contact')
def contact():
    return render_template('contact.html')


@application.route('/run-regressor', methods=['POST'])
def run_regressor():
    try:
        regression_algorithms = request.form.getlist('regression_algorithm')
        feature_selection_method = request.form.get('feature_selection_method')
        transformation_method = request.form.get('transformation_method')

        comparison_data = []
        best_model_data = None

        for regression_algorithm in regression_algorithms:
            result = get_regression_result(
                regression_algorithm, feature_selection_method, transformation_method)
            comparison_data.append({
                "Algorithm": result["Algorithm"],
                "R2": result["R2"],
                "RMSE": result["RMSE"]
            })

            if not best_model_data or float(result["R2"]) > float(best_model_data["R2"]):
                best_model_data = result

        # Ensure that bestModelData is sent even if only one algorithm is selected
        if not best_model_data and comparison_data:
            best_model_data = comparison_data[0]

        return jsonify({
            "comparisonTable": comparison_data,
            "bestModelData": best_model_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_regression_result(regression_algorithm, feature_selection_method, transformation_method):

    df_selected = df_original if feature_selection_method == "original" else (
        df_variance if feature_selection_method == "variance" else df_selKBest)

    target = df_transformed['sales']
    features = df_selected

    # generate an array of alpha values for ridge/lasso
    alphas = 10**np.linspace(5, -2, 15)

    if regression_algorithm == 'linear':
        r2, rmse, model, X_test, Y_test = perform_linear_regression(
            features, target, 0.25, 42, transformation_method)
    elif regression_algorithm == 'adaboost':
        r2, rmse, model, X_test, Y_test = perform_ada_regression(
            features, target, 0.25, 42, transformation_method)
    elif regression_algorithm == 'random_forest':
        r2, rmse, model, X_test, Y_test = perform_rf_regression(
            features, target, 0.25, 42, transformation_method)
    elif regression_algorithm == 'xgboost':
        r2, rmse, model, X_test, Y_test = perform_xgb_regression(
            features, target, 0.25, 42, transformation_method)
    elif regression_algorithm == 'decision_tree':
        r2, rmse, model, X_test, Y_test = perform_dt_regression(
            features, target, 0.25, 42, transformation_method)

    elif regression_algorithm == 'lasso':
        X_train, X_test, Y_train, Y_test = train_test_split(
            df_transformed.drop('sales', axis=1), target, test_size=0.2, random_state=42)
        # using 5-fold cross-validation
        lasso_cv = LassoCV(alphas=alphas, cv=5)
        lasso_cv.fit(X_train, Y_train)
        r2 = lasso_cv.score(X_test, Y_test)
        Y_pred = lasso_cv.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        model = lasso_cv
        best_alpha_lasso = lasso_cv.alpha_
    elif regression_algorithm == 'ridge':
        X_train, X_test, Y_train, Y_test = train_test_split(
            df_transformed.drop('sales', axis=1), target, test_size=0.2, random_state=42)
        # using 5-fold cross-validation
        ridge_cv = RidgeCV(alphas=alphas, cv=5)
        ridge_cv.fit(X_train, Y_train)
        r2 = ridge_cv.score(X_test, Y_test)
        Y_pred = ridge_cv.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        model = ridge_cv
        best_alpha_ridge = ridge_cv.alpha_
    else:
        return jsonify(result="Invalid algorithm choice!")

    regression_algorithm_proper = proper_case_linear_algorithm_name(
        regression_algorithm)

    result = {
        "Algorithm": regression_algorithm_proper,
        "R2": f"{r2:.4f}",
        "RMSE": f"{rmse:.4f}"
    }

    if regression_algorithm == 'lasso':
        result["Best Alpha"] = round(best_alpha_lasso, 2)
    elif regression_algorithm == 'ridge':
        result["Best Alpha"] = round(best_alpha_ridge, 2)

    Y_pred = model.predict(X_test)
    plot_data = {
        "actual": Y_test.tolist(),
        "predicted": Y_pred.tolist()
    }

    result["plotData"] = plot_data
    return result


@application.route('/run-classifier', methods=['POST'])
def run_classifier():
    try:
        # Extract form data
        classifier_names = request.form.getlist('classifier')
        classifier_params = {
            'kernel_param': request.form.get('kernel_param'),
            'C_param': request.form.get('C_param'),
            'gamma_param': request.form.get('gamma_param'),
            'max_depth_rf': request.form.get('max_depth_rf'),
            'max_depth_dt': request.form.get('max_depth_dt')
        }

        X_train, X_test, y_train, y_test = train_test_split(
            df_delaydetection, df['late_delivery_risk'], test_size=0.20, random_state=42)

        # initialize results list
        classifiers_results = []

        # run classification for each selected classifier
        for classifier_name in classifier_names:
            # Perform classification
            result, model_score, confusion_matrix_data, classification_rpt = perform_classification(
                classifier_name, classifier_params, X_train, X_test, y_train, y_test
            )
            classifiers_results.append({
                "Classifier": classifier_name,
                "Accuracy": model_score,
                "ConfusionMatrix": confusion_matrix_data.tolist(),
                "ClassificationReport": classification_rpt
            })

        # Sort classifiers by accuracy score in descending order
        classifiers_results.sort(key=lambda x: x["Accuracy"], reverse=True)

        # Get the best classifier's name
        best_classifier = classifiers_results[0]["Classifier"]

        # Return results outside of the for loop
        return jsonify({
            "comparisonTable": classifiers_results,
            "bestClassifier": best_classifier
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def perform_classification(classifier_name, classifier_params, X_train, X_test, y_train, y_test):
    classifiers = {
        'logistic_regression': LogisticRegression(),
        'k_neighbors': KNeighborsClassifier(3),
        'svc': SVC(
            kernel=classifier_params.get('kernel_param', 'rbf'),
            C=float(classifier_params.get('C_param', 1)),
            gamma=classifier_params.get('gamma_param', 'auto') if classifier_params.get(
                'kernel_param', 'rbf') == 'rbf' else 'scale',
        ),
        'decision_tree': DecisionTreeClassifier(max_depth=int(classifier_params.get('max_depth_dt', None))),
        'naive_bayes': GaussianNB(),
        'xgboost': XGBClassifier(),
        'catboost': CatBoostClassifier()
    }

    classifier = classifiers.get(classifier_name)
    if not classifier:
        raise ValueError(f"Classifier {classifier_name} is not supported")

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    model_score = pipeline.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    create_confusion_matrix_heatmap(cm, classifier_name)

    return y_pred, model_score, cm, cr


if __name__ == '__main__':
    application.run(port=5000, debug=False)
