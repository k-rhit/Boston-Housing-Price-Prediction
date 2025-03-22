import pandas as pd
from src.data_loader import load_data, load_config
from src.eda import *
from src.feature_engineering import *
from src.model_training import train_select_model

def main():
    # Load configuration
    config = load_config("config/config.yaml")

    # Load dataset
    filepath = config["data"]["dataset_path"]
    df = load_data(filepath)
    print(df.head())

    # Perform EDA

    dataset_overview(df)
    flag = missing_values(df)
    # Univariate analysis
    plot_histogram(df)
    # Bivariate Correlation Analysis
    plot_correlation_heatmap(df)
    # Visualize and detect outliers
    plot_boxplot(df)

    # Feature Engineering
    # Handle missing values if present
    if flag == True:
        handling_missing_values(df)

    # Detect and handle outliers
    df = handling_outliers(df)
    # Power transform skewed features
    df = tranform_skewed_feature(df)
    # Feature selection
    df = feature_selection(df, "PRICE")  
    
    # Save processed data
    processed_filepath = config["processed_data"]["processed_dataset_path"]
    df.to_csv(processed_filepath, index = False)
    print(f"\nProcessed datasets saved!")

    # Path to save model
    modelpath = config["model"]["model_path"]

    # Train model
    model_path = train_select_model(processed_filepath, modelpath)

if __name__ == "__main__":
    main()
