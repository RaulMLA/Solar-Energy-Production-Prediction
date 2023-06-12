# Solar Energy Production Prediction
The purpose of this project is to work with different machine learning methods and hyperparameter tuning/optimization (HPO). Additionally, it aims to practice the entire process: determining the best method for a given dataset (model selection, including hyperparameter tuning), estimating the future performance of the best method (model evaluation), and building the final model to make predictions on new data (model usage).

Nowadays, advanced countries' power grids are increasingly reliant on non-dispatchable renewable energy sources, primarily wind and solar. However, integrating these energy sources into the power grid requires predicting the amount of energy to be generated 24 hours in advance, so that power plants connected to the grid can plan and prepare to meet supply and demand for the following day.

This is not a problem for traditional energy sources (gas, oil, hydroelectric power, etc.) because they can be generated (managed) at will (by burning more gas, for example). But solar and wind energies are not under the control of the energy operator because they depend on the weather. The only alternative is to predict them as accurately as possible. This can be achieved to some extent through weather forecasting, which involves simulating the atmosphere using physical-mathematical models. The Global Forecast System (GFS) in the US, also known as the Global Ensemble Forecast System (GEFS), and the European Centre for Medium-Range Weather Forecasts (ECMWF) are two of the most important numerical weather prediction (NWP) models.

However, while NWP models are very good at predicting variables like "average downward longwave radiation flux at the surface" related to solar radiation, the relationship between those variables and the actual electricity produced in solar plants is not straightforward. Machine learning models can be used for this latter task.

In particular, we will use meteorological variables predicted by GFS as input attributes to a machine learning model capable of estimating how much solar energy will be produced in solar plants in the state of Oklahoma.

# Contents
- [Methods](#methods)
  - [Basic Methods](#basic-methods)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Advanced Methods](#advanced-methods)
  - [Best Method Selection and Final Model](#best-method-selection-and-final-model)
- [Authors](#authors)

# Methods

In this project, we will explore different machine learning methods for predicting solar energy production. We will evaluate the performance of each method using the following evaluation metrics: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Additionally, we will measure the training time for each method.

Note that methods explored are contained in the *AA_P1_Cuaderno_1.ipynb* file and *AA_P1_Cuaderno_2.ipynb* file contains the code for the final model. All the explanations and conclusions are in those files.

## Basic Methods
In this section, we will consider the following basic methods for predicting solar energy production:

- ``KNN (K-Nearest Neighbors)``:
KNN is a non-parametric method that makes predictions based on the similarity between new data points and existing labeled data. It works by finding the K nearest neighbors to a given data point and predicting its output based on the majority vote or averaging the outputs of those neighbors. KNN is simple to implement and can capture complex relationships in the data. We will explore its performance in predicting solar energy production.

- ``Regression Trees``:
Regression trees are decision tree-based models that recursively partition the input space into regions based on the values of input features. Each region represents a subset of the data with similar characteristics. Regression trees are capable of handling both categorical and continuous input features. They are intuitive, easy to interpret, and can capture non-linear relationships in the data. We will assess the performance of regression trees in predicting solar energy production.

- ``Linear Regression``:
Linear regression is a simple and widely used method for modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the predictors and the target variable and estimates the coefficients that best fit the data. Linear regression is computationally efficient and provides interpretable coefficients. We will evaluate its performance in predicting solar energy production.

We will evaluate these models using their default hyperparameters. The evaluation will include calculating the RMSE and MAE metrics, as well as measuring the training time.

After loading the necessary datasets, we will perform the following steps for each basic method:
1. Evaluate the model using default hyperparameters.
2. Adjust the most important hyperparameters for each method and evaluate their performance.

We will draw conclusions based on the results obtained. Some possible questions to consider are:
- Which method performs the best?
- Which basic machine learning method is the fastest?
- Are the results better than trivial/naive/baseline regressors?
- Does hyperparameter tuning improve the performance compared to default values? 
- Is there a trade-off between execution time and performance improvement?

## Dimensionality Reduction
In this part, we will explore the possibility of reducing the dimensionality of the problem. We can consider various approaches to achieve this, not necessarily limited to standard techniques. The goal is to reduce the number of attributes in the data while maintaining or improving the prediction results.

## Advanced Methods
In this section, we will explore two advanced machine learning methods for predicting solar energy production:

- ``Support Vector Machines (SVMs)``:
Support Vector Machines are powerful supervised learning models that can be used for both classification and regression tasks. SVMs aim to find an optimal hyperplane that separates the data points of different classes or predicts the continuous target variable with the maximum margin. SVMs are effective in handling high-dimensional data and can capture complex relationships through the use of kernel functions. We will evaluate the performance of SVMs in predicting solar energy production and assess their ability to handle the specific characteristics of the dataset.

- ``Random Forests``:
Random Forests are ensemble learning models that combine multiple decision trees to make predictions. Each tree in the random forest is trained on a random subset of the data, and the final prediction is obtained through averaging or voting among the individual tree predictions. Random Forests are robust against overfitting, handle both numerical and categorical features, and provide estimates of attribute importance. We will explore the performance of Random Forests in predicting solar energy production and interpret the importance of the input attributes using the techniques provided by the model.

We will initially evaluate these models using their default hyperparameters. Then, we will adjust the most important hyperparameters for each method and evaluate their performance. For methods that provide attribute importance scores, we will interpret the significance of the attributes.

We will draw conclusions based on the findings up to this point.

## Best Method Selection and Final Model
In this part, we will select the best-performing method based on the evaluations conducted in the previous sections. We will use the test partition to assess the performance of the selected method, which serves as an estimation of how the model would perform in a competition setting.

Once the best method is determined, we will train the final model using the entire dataset. The final model will be saved in a file named "modelo_final.pkl".

Finally, we will utilize the final model to make predictions on the competition dataset (comp). The predictions will be saved in a file named "predicciones.csv".

# Authors
- [Raúl Manzanero López-Aguado](https://github.com/RaulMLA)
- [Adrián Sánchez del Cerro](https://github.com/adrisdc)
