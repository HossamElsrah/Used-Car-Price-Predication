# Used-Car-Price-Predication

Project Overview
The Used Cars Price Prediction project is designed to provide insights into the pricing of used cars based on various features. This machine learning application leverages different regression models—Linear Regression, Ridge Regression, and Polynomial Regression—to predict the prices of used cars with a high degree of accuracy. The primary goal of this project is to assist potential buyers and sellers in making informed decisions by predicting the fair market value of used vehicles.

Dataset
The dataset used for this project contains various attributes of used cars, including but not limited to:

Brand: The manufacturer of the car.
Model: The specific model of the car.
Year: The year of manufacture.
Kilometers Driven: The total distance the car has been driven.
Seats: The seating capacity of the car.
Location: The geographical area where the car is being sold.
Price: The target variable representing the market price of the car.
The dataset is preprocessed to handle missing values, encode categorical variables, and scale numerical features where necessary. This ensures that the models can learn effectively from the data.

Models Used
1. Linear Regression
Linear Regression is a foundational machine learning algorithm used to establish a relationship between the independent variables (features) and the dependent variable (price). In this project, Linear Regression serves as a baseline model to predict the price of used cars based on their features.

2. Ridge Regression
Ridge Regression is an extension of Linear Regression that incorporates L2 regularization. This technique helps prevent overfitting by penalizing large coefficients, making it particularly useful when dealing with multicollinearity or when the number of features is high relative to the number of observations. Ridge Regression provides improved generalization on unseen data compared to standard Linear Regression.

3. Polynomial Regression
Polynomial Regression allows us to model the relationship between the independent variables and the dependent variable as a polynomial function. This approach is beneficial when the relationship is not linear. By transforming the input features into polynomial features, we can capture more complex patterns in the data, leading to better predictions in certain scenarios.

Implementation
The project is implemented using Python and leverages popular libraries such as:

Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Scikit-learn: For building and evaluating the regression models.
Streamlit: For creating an interactive web application to visualize predictions and user inputs.
Steps in the Implementation:
Data Loading: The dataset is loaded from a CSV file.
Data Preprocessing: Missing values are handled, categorical features are encoded, and numerical features are scaled.
Model Training: The Linear Regression, Ridge Regression, and Polynomial Regression models are trained on the training dataset.
Model Evaluation: Each model is evaluated using appropriate metrics (e.g., Mean Absolute Error, R-squared) to determine performance.
Web Application: An interactive Streamlit app is developed, allowing users to input car features and receive price predictions based on the trained models.
