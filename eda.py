import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def stats(df: pd.DataFrame) -> pd.DataFrame:
    num_df = df.select_dtypes(include='number')
    stats_df = pd.DataFrame({
        'Średnia': num_df.mean(),
        'Mediana': num_df.median(),
        'Wsp. zmienności': num_df.std(ddof=0) / num_df.mean(),
        'Max': num_df.max(),
        'Min': num_df.min(),
        'Rozstęp': num_df.max() - num_df.min(),
    }).T

    return stats_df


def df_compare(df1: pd.DataFrame, df2: pd.DataFrame, title1: str = 'Tabela 1', title2: str = 'Tabela 2') -> pd.DataFrame:
    return pd.concat([df1, df2], keys=[title1, title2])


def box_plots(df: pd.DataFrame, columns: list) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(8, 5))
    axes = axes.flatten()
    for ax, col in zip(axes, columns):
        sns.boxplot(y=df[col], color='lightblue', ax=ax)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel('')
        ax.tick_params(labelsize=7)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle('Wykresy pudełkowe poszczególnych zmiennych', fontsize=12)
    fig.tight_layout()
    plt.show()


def correlation_matrix_heatmap(df: pd.DataFrame, columns: list) -> None:
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', square=True)
    plt.title('Macierz korelacji zmiennych')
    plt.show()


def outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    outlier_info = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'liczba_outlierów': len(outliers),
            'regiony_wystepowania_outlierow': outliers['Województwo'].tolist()
        }

    return pd.DataFrame(outlier_info).T