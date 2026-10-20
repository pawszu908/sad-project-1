import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def normalize_data(df, stimulants, all_vars):
    x = df[all_vars].copy()
    z_std = pd.DataFrame(index=df.index)

    for col in all_vars:
        mean_v = x[col].mean()
        std_v = x[col].std(ddof=1)
        if col in stimulants:
            z_std[col] = (x[col] - mean_v) / std_v
        else:
            z_std[col] = (mean_v - x[col]) / std_v

    z_unit = pd.DataFrame(index=df.index)
    for col in all_vars:
        min_v = x[col].min()
        max_v = x[col].max()
        if col in stimulants:
            z_unit[col] = (x[col] - min_v) / (max_v - min_v)
        else:
            z_unit[col] = (max_v - x[col]) / (max_v - min_v)

    return z_std, z_unit

def calculate_synthetic_variables(df, z_std, z_unit):
    q_r1 = z_std.mean(axis=1)
    q_r2 = z_unit.mean(axis=1)

    z_0 = z_std.max(axis=0)
    d_i0 = np.sqrt(((z_std - z_0) ** 2).sum(axis=1))
    d_bar = d_i0.mean()
    s_d = d_i0.std(ddof=0)
    d_0 = d_bar + 2 * s_d
    q_r5 = 1 - (d_i0 / d_0)

    z_plus = z_std.max(axis=0)
    z_minus = z_std.min(axis=0)
    d_plus = np.sqrt(((z_std - z_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((z_std - z_minus) ** 2).sum(axis=1))
    q_r6 = d_minus / (d_plus + d_minus)

    df_res = pd.DataFrame({
        "Województwo": df["Województwo"].values,
        "q_R1": q_r1,
        "q_R2": q_r2,
        "q_R5": q_r5,
        "q_R6": q_r6,
    })

    df_res["R1"] = df_res["q_R1"].rank(ascending=False, method="min").astype(int)
    df_res["R2"] = df_res["q_R2"].rank(ascending=False, method="min").astype(int)
    df_res["R5"] = df_res["q_R5"].rank(ascending=False, method="min").astype(int)
    df_res["R6"] = df_res["q_R6"].rank(ascending=False, method="min").astype(int)

    return df_res


def evaluate_similarity(df_res, n_elements):
    ranks_matrix = df_res[["R1", "R2", "R5", "R6"]].values
    v = ranks_matrix.shape[1]
    u_matrix = np.zeros((v, v))
    z_n = 1 if n_elements % 2 != 0 else 0
    denom = n_elements * (n_elements ** 2 - 1) - z_n

    for p in range(v):
        for q in range(v):
            sum_sq_diff = np.sum((ranks_matrix[:, p] - ranks_matrix[:, q]) ** 2)
            u_matrix[p, q] = 1 - ((2 * sum_sq_diff) / denom)

    df_u = pd.DataFrame(u_matrix, index=["R1", "R2", "R5", "R6"], columns=["R1", "R2", "R5", "R6"])
    u_bar_p = (df_u.sum(axis=1) - 1.0) / (v - 1)
    df_u_bar = pd.DataFrame({"Ranking": ["R1", "R2", "R5", "R6"], "u_bar_p": u_bar_p.values})

    best_ranking_code = df_u_bar.sort_values(by="u_bar_p", ascending=False).iloc[0]["Ranking"]

    return df_u, df_u_bar, best_ranking_code

def assign_classes(series):
    m = series.mean()
    s = series.std(ddof=1)
    c1_bound = m + 0.8416 * s
    c2_bound = m + 0.2533 * s
    c3_bound = m - 0.2533 * s
    c4_bound = m - 0.8416 * s

    classes = []
    for val in series:
        if val >= c1_bound:
            classes.append("Klasa I (bardzo wysoki)")
        elif val >= c2_bound:
            classes.append("Klasa II (wysoki)")
        elif val >= c3_bound:
            classes.append("Klasa III (średni)")
        elif val >= c4_bound:
            classes.append("Klasa IV (niski)")
        else:
            classes.append("Klasa V (bardzo niski)")
    return classes

def compare_with_historical(df_res, best_ranking_code, ranks_historical):
    df_comp = df_res[["Województwo", best_ranking_code]].copy()
    df_comp.rename(columns={best_ranking_code: "R_2023"}, inplace=True)
    df_comp["R_2014"] = df_comp["Województwo"].map(ranks_historical)
    df_comp["Zmiana_pozycji"] = df_comp["R_2014"] - df_comp["R_2023"]
    return df_comp


def plot_ranking_comparison(df_comp):
    df_long = df_comp.melt(
        id_vars=["Województwo", "Zmiana_pozycji"],
        value_vars=["R_2014", "R_2023"],
        var_name="Year",
        value_name="Rank"
    )

    df_long["Year"] = df_long["Year"].map({
        "R_2014": "2014",
        "R_2023": "2023"
    })

    color_map = {}
    for _, row in df_comp.iterrows():
        if row["Zmiana_pozycji"] > 0:
            color_map[row["Województwo"]] = "lightgreen"
        elif row["Zmiana_pozycji"] < 0:
            color_map[row["Województwo"]] = "firebrick"
        else:
            color_map[row["Województwo"]] = "slateblue"

    plt.figure(figsize=(11, 8.5))

    ax = sns.pointplot(
        data=df_long,
        x="Year",
        y="Rank",
        hue="Województwo",
        palette=color_map,
        markers="o",
        linestyles="-",
        linewidth=2.5,
        markersize=8
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    for _, row in df_comp.iterrows():
        woj = row["Województwo"]
        y2014 = row["R_2014"]
        y2023 = row["R_2023"]

        plt.text(
            -0.06, y2014, f"{woj} ({y2014})",
            ha="right", va="center", fontsize=9.5,
            fontweight="normal"
        )
        plt.text(
            1.06, y2023, f"{woj} ({y2023})",
            ha="left", va="center", fontsize=9.5,
            fontweight="normal"
        )

    plt.gca().invert_yaxis()
    plt.yticks(range(1, 17), [""] * 16)

    plt.xlabel("")
    plt.ylabel("")
    plt.title("Zmiana pozycji województw w rankingu potencjału rolnictwa", fontsize=13, pad=18, fontweight="bold")

    plt.xlim(-0.8, 1.8)
    plt.ylim(16.5, 0.5)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()