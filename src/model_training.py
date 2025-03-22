import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha = 1.0),
    "Lasso Regression": Lasso(alpha = 0.1),
    "Decision Tree": DecisionTreeRegressor(random_state = 42),
    "Random Forest": RandomForestRegressor(random_state = 42),
}

# Hyperparameter grids for tuning
param_grids = {
    "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
    "Lasso Regression": {"alpha": [0.001, 0.01, 0.1, 1]},
    "Decision Tree": {"max_depth": [10, 20, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]},
    "Random Forest": {"n_estimators": [100, 200, 300],"max_depth":[10, 20, None], "min_samples_split" : [2, 5, 10]},
}

def model_evaluation(model, x_train, x_test, y_train, y_test):
    """Train and evaluate model on testing data"""
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE":rmse, "R2 Score": r2}

def hyperparameter_tuning(best_model_name, best_model, x_train, y_train):
    """Hyperparameter tuning using GridSearchCV"""
    print(f"Performing hyperparameter tuning for {best_model_name}")
    gscv = GridSearchCV(best_model, param_grids[best_model_name], cv = 5, scoring = "r2", n_jobs = -1)
    gscv.fit(x_train, y_train)
    print(f"Best parameters for {best_model_name}: {gscv.best_params_}")
    return gscv.best_estimator_

def train_select_model(file_path, model_path):
    """Train models, compare performance and select best one"""
    # Load preprocessed data
    df = pd.read_csv(file_path)
    # splitting target variable
    x = df.drop("PRICE", axis = 1)
    y = df["PRICE"]
    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    # Store model performance
    model_performance = {}    
    for name, model in models.items():
        print(f"Training {name} model...")
        scores = model_evaluation(model, x_train, x_test, y_train, y_test)
        model_performance[name] = scores
    perf_df = pd.DataFrame(model_performance).T
    print("\n Model Evaluation Comparison:\n", perf_df)

    # Select best model based on R2 Score
    best_model_name = max(model_performance, key = lambda k : model_performance[k]["R2 Score"])
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print(f"Performance: {model_performance[best_model_name]}")

    # Hyperparameter tuning best model
    best_model_tuned = hyperparameter_tuning(best_model_name, best_model, x_train, y_train)
    final_scores = model_evaluation(best_model_tuned, x_train, x_test, y_train, y_test)
    print(f"\n{best_model_name} performance after tuning\n")
    print(f"MAE: {final_scores["MAE"]}")
    print(f"MSE: {final_scores["MSE"]}")
    print(f"RMSE: {final_scores["RMSE"]}")
    print(f"R2 Score: {final_scores["R2 Score"]}")

    # Save the best model using pickle
    with open (model_path, "wb") as f:
        pickle.dump(best_model_tuned, f)
    print(f"\nModel saved!")

    return model_path

