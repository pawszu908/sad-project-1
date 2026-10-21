import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def prepare_clustering_data(df, z_std):
    x = z_std.values.astype(float)
    names = df["Województwo"].values

    df_indexed = z_std.copy()
    df_indexed.insert(0, "Województwo", names)

    return x, names, df_indexed

def hierarchical_clustering(x):
    return linkage(x, method="ward", metric="euclidean")

def select_k_hierarchical(z, x, k_range=range(2, 9)):
    rows = []
    for k in k_range:
        labels = cut_tree(z, n_clusters=[int(k)]).ravel()
        n_lab = len(np.unique(labels))
        if not (1 < n_lab < len(x)):
            rows.append({"k": int(k), "G1": np.nan, "S": np.nan, "DB": np.nan})
            continue
        rows.append({
            "k": int(k),
            "G1": round(float(calinski_harabasz_score(x, labels)), 3),
            "S":  round(float(silhouette_score(x, labels)), 3),
            "DB": round(float(davies_bouldin_score(x, labels)), 3),
        })
    return pd.DataFrame(rows)

def plot_dendrogram(z, names, k=None):
    color_threshold = None
    if k is not None and 1 < k < len(names):
        color_threshold = (z[-k, 2] + z[-k + 1, 2]) / 2.0

    plt.figure(figsize=(11, 5.5))
    dendrogram(z, labels=names, leaf_rotation=90, leaf_font_size=9,
               color_threshold=color_threshold)
    plt.title("Dendrogram — grupowanie hierarchiczne (Ward)")
    plt.xlabel("Województwo")
    plt.ylabel("Odległość wiązania (Ward)")
    plt.tight_layout()
    plt.show()

def plot_merge_heights(z):
    heights = z[:, 2]
    steps = np.arange(1, len(heights) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, heights, "o-", color="navy", markersize=4)
    plt.xlabel("Krok wiązania")
    plt.ylabel("Odległość wiązania")
    plt.title("Wykres odległości wiązania względem etapów wiązania (Timm 2002)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({"krok": steps, "odleglosc": np.round(heights, 3)})

from sklearn.decomposition import PCA

def plot_pca_clusters(x, labels, names):
    pca = PCA(n_components=2)
    y = pca.fit_transform(x)
    ev = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
    for i, g in enumerate(sorted(np.unique(labels))):
        m = labels == g
        plt.scatter(y[m, 0], y[m, 1], s=130, color=palette[i],
                    label=f"Klaster {g}", edgecolor="black")
        for j in np.where(m)[0]:
            plt.annotate(names[j], (y[j, 0], y[j, 1]),
                         fontsize=8, xytext=(5, 5), textcoords="offset points")
    plt.xlabel(f"PC1 ({ev[0]:.1f}%)")
    plt.ylabel(f"PC2 ({ev[1]:.1f}%)")
    plt.title("Wizualizacja skupień w przestrzeni PCA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return pca

def plot_cluster_indices(df_k):
    specs = [
        ("G1", "Calinski-Harabasz (max)", "seagreen",   "max"),
        ("S",  "Silhouette (max)",        "firebrick",  "max"),
        ("DB", "Davies-Bouldin (min)",    "darkorange", "min"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, subtitle, color, direction) in zip(axes, specs):
        ax.plot(df_k["k"], df_k[col], "o-", color=color, markersize=7)

        if direction == "max":
            best_k = int(df_k.loc[df_k[col].idxmax(), "k"])
        else:
            best_k = int(df_k.loc[df_k[col].idxmin(), "k"])
        best_val = float(df_k.loc[df_k["k"] == best_k, col].iloc[0])

        ax.axvline(best_k, color="gray", linestyle="--", alpha=0.5)
        ax.scatter([best_k], [best_val], s=200, facecolor="none",
                   edgecolor="red", linewidth=2, zorder=5)
        ax.annotate(f"k = {best_k}", (best_k, best_val),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)

        ax.set_xlabel("Liczba skupień (k)")
        ax.set_ylabel(col)
        ax.set_title(subtitle)
        ax.grid(alpha=0.3)

    plt.suptitle("Wybór liczby skupień — indeksy jakości klasyfikacji", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def summarize_best_k(df_k):
    votes = {
        "G1": int(df_k.loc[df_k["G1"].idxmax(), "k"]),
        "S":  int(df_k.loc[df_k["S"].idxmax(),  "k"]),
        "DB": int(df_k.loc[df_k["DB"].idxmin(), "k"]),
    }
    df_votes = pd.DataFrame({"indeks": list(votes),
                             "wskazane_k": list(votes.values())})
    majority_k = int(df_votes["wskazane_k"].mode().iloc[0])
    return df_votes, majority_k

def profile_groups(df, labels, variables):
    df_p = df[["Województwo"] + variables].copy()
    df_p["Klaster"] = labels
    return (df_p.groupby("Klaster")[variables]
                 .agg(["mean", "std", "min", "max", "count"])
                 .round(2))


def list_group_members(names, labels):
    return {int(g): sorted(names[labels == g].tolist())
            for g in np.unique(labels)}