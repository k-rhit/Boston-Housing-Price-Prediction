import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor 

def handling_missing_values(df):
    """Fill missing values using Median"""
    df.fillna(df.median(), inplace = True)
    print("Missing values filled using Mean")

def handling_outliers(df):
    """Detect and cap outliers using IQR method"""
    print("\nNumber of outliers:")
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound)| (df[col] > upper_bound)]
        print(f"\t{col} : {len(outliers)}")
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    print("\nOutliers are capped")
    return df

def tranform_skewed_feature(df):
    """Transform skewed features using Power Transform"""
    transformer = PowerTransformer()
    df[df.columns] = transformer.fit_transform(df)
    print("\nFeatures are transformed using power transform")
    return df

def feature_selection(df, target, n_feature = 10):
    """Recursive feature elimination using Random Forest"""
    x = df.drop(target, axis = 1)
    y = df[target]
    model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rfe = RFE(model, n_features_to_select = n_feature)
    x_rfe = rfe.fit_transform(x,y)
    selected_columns = x.columns[rfe.support_]
    print(f"Selected Features using Recursive Feature Elimination: {selected_columns}")
    df = df[selected_columns].join(y)
    return df