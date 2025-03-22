import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def dataset_overview(df):
    """
    Print dataset structure and summary statistics
    """
    print("\nDataset Information:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nSample Data:")
    print(df.sample(5))

def missing_values(df):
    """Check Missing Values"""
    missing = df.isnull().sum()
    flag = True
    if missing.sum() == 0:
        print("There are no missing values in the dataset")
        flag = False
    else:
        print("\nMissing Values:")
        print(missing[missing > 0])
    return flag
    
def plot_histogram(df):
    """Plot histogram of a feature for univariate analysis"""
    for col in df.columns:
        plt.figure(figsize = (8,5))
        sns.histplot(df[col], bins = 30, kde = True)
        plt.title(f"Distribution of {col}")
        plt.show()

def plot_correlation_heatmap(df):
    """Check correlation among features"""
    plt.figure(figsize = (10,6))
    sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(df)
    plt.show()

def plot_boxplot(df):
    """Plot boxplot of a feature to visualize outliers"""
    # for col in df.select_dtypes(include = ["float64", "int64"]).columns: 
    for col in df.columns:
        plt.figure(figsize = (6,4))
        sns.boxplot(x = df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()
