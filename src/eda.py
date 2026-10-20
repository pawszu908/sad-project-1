import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic statistics for each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame containing statistics for each column.
    """
    num_df = df.select_dtypes(include='number')
    stats_df = num_df.describe()
    stats_df.loc['coefficient_of_variation'] = num_df.std() / num_df.mean()
    stats_df.loc['range'] = num_df.max() - num_df.min()
    stats_df.loc['skewness'] = num_df.skew()
    stats_df = stats_df.round(2)

    return stats_df

def box_plots(df: pd.DataFrame, columns: list) -> None:
    """
    Generate box plots for specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to generate box plots for.

    Returns:
    None
    """

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col], color='lightblue')
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)
        plt.show()

def correlation_matrix_heatmap(df: pd.DataFrame, columns: list) -> None:
    """
    Generate a heatmap of the correlation matrix for the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to include in the correlation matrix.

    Returns:
    None
    """
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', square=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()


def outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Identify outliers in specified columns of the DataFrame using the IQR method.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to check for outliers.

    Returns:
    pd.DataFrame: DataFrame containing outlier information for each specified column with the vovoideship names included.
    """
    outlier_info = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'num_outliers': len(outliers),
            'outlier_regions': outliers['Województwo'].tolist()
        }

    return pd.DataFrame(outlier_info).T