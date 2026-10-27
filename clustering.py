import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

def prepare_clustering_data(df, z_std):
    x = z_std.values.astype(float)
    names = df["Województwo"].values

    df_indexed = z_std.copy()
    df_indexed.insert(0, "Województwo", names)

    return x, names, df_indexed

def hierarchical_clustering(x):
    return linkage(x, method="ward", metric="euclidean")

def plot_k_hierarchical(z, x, k_range=range(2, 9)):
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

def plot_dendrogram(z, names):
    plt.figure(figsize=(11, 5.5))
    dendrogram(z, labels=names, leaf_rotation=90, leaf_font_size=9, color_threshold=None)
    plt.title("Dendrogram - grupowanie hierarchiczne (Ward)")
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

def plot_cluster_indices(df_k):
    specs = [
        ("G1", "Calinski-Harabasz (max)", "seagreen",   "max"),
        ("S",  "Silhouette (max)",        "firebrick",  "max"),
        ("DB", "Davies-Bouldin (min)",    "darkorange", "min"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, subtitle, color, direction) in zip(axes, specs):
        ax.plot(df_k["k"], df_k[col], "o-", color=color, markersize=7)
        ax.set_xlabel("Liczba skupień (k)")
        ax.set_ylabel(col)
        ax.set_title(subtitle)
        ax.grid(alpha=0.3)

    plt.suptitle("Krzywe indeksów jakości klasyfikacji", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

def profile_groups(df, labels, variables):
    df_p = df[["Województwo"] + variables].copy()
    df_p["Klaster"] = labels

    means = df_p.groupby("Klaster")[variables].mean().round(2)
    stds = df_p.groupby("Klaster")[variables].std().round(2).fillna(0)

    result = pd.DataFrame(index=means.index, columns=variables)
    for var in variables:
        result[var] = means[var].astype(str) + " (" + stds[var].astype(str) + ")"

    overall_mean = df[variables].mean().round(2)
    overall_std = df[variables].std().round(2).fillna(0)
    result.loc["Ogółem"] = overall_mean.astype(str) + " (" + overall_std.astype(str) + ")"

    result = result.T
    result.index.name = "Zmienna"
    result.columns.name = "Klaster"

    return result


def list_group_members(names, labels):
    return {int(g): sorted(names[labels == g].tolist())
            for g in np.unique(labels)}


def kmeans_indices(x, k_range=range(2, 9)):
    rows = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=100, random_state=42)
        labels = kmeans.fit_predict(x)
        n_lab = len(np.unique(labels))
        if not (1 < n_lab < len(x)):
            rows.append({"k": int(k), "G1": np.nan, "S": np.nan, "DB": np.nan, "inertia": np.nan})
            continue
        rows.append({
            "k": int(k),
            "G1": round(float(calinski_harabasz_score(x, labels)), 3),
            "S":  round(float(silhouette_score(x, labels)), 3),
            "DB": round(float(davies_bouldin_score(x, labels)), 3),
            "inertia": round(float(kmeans.inertia_), 3)
        })
    return pd.DataFrame(rows)


def plot_elbow(df_k):
    plt.figure(figsize=(10, 5))
    plt.plot(df_k["k"], df_k["inertia"], "o-", color="navy", markersize=7)
    plt.xlabel("Liczba skupień (k)")
    plt.ylabel("Inercja (suma kwadratów odległości do centroidów)")
    plt.title("Wykres inercji względem liczby skupień (metoda łokcia)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_indices_comparison(df_a, df_b, label_a="Ward", label_b="k-means"):
    specs = [("G1", "Calinski-Harabasz (max)"),
             ("S", "Silhouette (max)"),
             ("DB", "Davies-Bouldin (min)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title) in zip(axes, specs):
        ax.plot(df_a["k"], df_a[col], "o-", label=label_a, markersize=7)
        ax.plot(df_b["k"], df_b[col], "s--", label=label_b, markersize=6)
        ax.set_title(title)
        ax.set_xlabel("Liczba skupień (k)")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.suptitle("Porównanie wskaźników jakości: Ward vs k-means", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

