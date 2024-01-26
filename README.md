<p align="center">
    <img src='static/img/logo.png' width=200 class="center">
    <h2 align="center">Full-Stack Web Application for Supply Chain Management</h2>
</p>

### Overview

This project is a full-stack web application that enhances supply chain management through the use of machine learning models. It integrates Python and its machine learning libraries for backend processing, and employs Flask, along with HTML, CSS, and JavaScript for the frontend. This combination provides an intuitive interface for user interaction and facilitates data-driven decision-making in supply chain operations.

### Features

#### Front End

- **Interactive Interface**: Developed with HTML, CSS, and JavaScript for effective user engagement.
- **Structure and Navigation**: Segmented into multiple pages for streamlined user interaction.
  - **Index Page**: Displays dashboards and visualizations for Business Intelligence.
    <p align="center"> <img src='static/img/index_page.png' width=600 class="center"></p>
  - **Sales Forecasting**: Users can interact with machine learning models for sales predictions.
    <p align="center"> <img src='static/img/sales_forecasting_page.png' width=600 class="center"></p>
  - **Delay Detection**: Detects late deliveries to optimize logistic.
    <p align="center"> <img src='static/img/delay_detection_page.png' width=600 class="center"></p>
  - **Contact Page**: Provides information and a communication channel.
    <p align="center"> <img src='static/img/contact_page.png' width=600 class="center"></p>

### Back End

- **Flask Framework**: Powers the application's back-end functionalities.
- **Model Execution Route**: Handles POST requests for machine learning predictions.
- **Error Handling**: Robust mechanisms for a smooth user experience.

## Sales Forecasting - Linear Regression Approach

- **Data Processing**: Involves cleaning, dimensionality reduction, and feature creation.
- **Feature Selection**: Uses methods like Variance Threshold and K-Best.
- **Model Execution**: Allows comparison of regression algorithms and feature selections.
 <p align="center"> <img src='static/img/result_of_multiple_algorithm_sales_forecasting.png' width=600 class="center"></p>
 
 <p align="center"> <img src='static/img/descriptive_explanation_sales_forecasting.png' width=600 class="center"></p>

  <p align="center"> <img src='static/img/scatterplot_sales_forecasting.png' width=600 class="center"></p>

## Delay Detection - Classification Approach

- **Data Transformation**: Includes label encoding and feature selection.
- **Model Execution**: Supports various classifiers with hyperparameter customization.
  
 <p align="center"> <img src='static/img/result_of_multiple_algorithm_delay_detection.png' width=600 class="center"></p>
 
 <p align="center"> <img src='static/img/cm_delay_detection.png' width=600 class="center"></p>

 <p align="center"> <img src='static/img/cr_delay_detection.png' width=600 class="center"></p>
  
## Dashboard - Supply Chain Performance Visualization

- **Visualization Generation**: Utilizes Matplotlib and Seaborn for dynamic charts and graphs.
- **Interactive Features**: Allows real-time editing and updating of analysis text.

 <p align="center"> <img src='static/img/scm_dashboard.png' width=600 class="center"></p>

 <p align="center"> <img src='static/img/editable_texts.png' width=600 class="center"></p>

## Lessons Learned and Future Work

- **Full-Stack Development Experience**: Enhanced skills in JavaScript, CSS, HTML, and machine learning integration.
- **Flask Integration and Debugging**: Learned integration of various technologies into a cohesive application.
- **Future Work**: Plans include deployment options and feature expansion like fraud detection.

## Conclusion

This project demonstrates the integration of machine learning in supply chain management, emphasizing data-driven decision-making and efficiency. It stands as a testament to the power of combining different technologies and continuous learning in the tech and data science fields.
