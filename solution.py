#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World bank data clustering
@author: Muhammad Arafat Azam (id: 21087019)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from textwrap import wrap

YEARS = [f'{i}' for i in range(1990, 2020, 4)]


def normalize(data: pd.DataFrame) -> pd.DataFrame:
    df = data
    for col in data.columns:
        df[col] = df[col]/df[col].abs().max()
    return df


def scatter_matrix(data: pd.DataFrame, do_normalize: bool = True):
    df = data[YEARS]

    if do_normalize:
        df = normalize(data)

    # Unstacking 2nd index (Countries) into columns and then transposing
    df = df.unstack(level=1).T

    df.columns = ['\n'.join(wrap(x, 20)) for x in df.columns]
    
    axs = pd.plotting.scatter_matrix(df, figsize=(8, 8), s=5, alpha=0.8)
    
    for ax in axs.flatten():
        ax.xaxis.label.set_rotation(45)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
        
    plt.suptitle('Scatter Matrix')
    plt.tight_layout()
    plt.show()


def corr_heat_map(data: pd.DataFrame):
    # Unstacking 2nd index (Countries) into columns and then transposing
    df = data.unstack(level=1).T

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(corr, interpolation='nearest', cmap='RdYlGn')
    fig.colorbar(im, orientation='vertical', fraction=0.05)
    ax.set_xticklabels(['']+["\n".join(wrap(x, 30))
                       for x in corr.columns.to_list()], rotation=90, fontsize=8)
    ax.set_yticklabels(['']+["\n".join(wrap(x, 30))
                       for x in corr.index.to_list()], rotation=0, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, round(corr.to_numpy()[i, j], 2),
                           ha="center", va="center", color="black")
    plt.title('Correlation Heat Map')
    plt.tight_layout()
    plt.show()

def kmean_elbow_plot(data: pd.DataFrame):
    w = []
    for i in range(10):
        kmeans = KMeans(n_clusters=i+1)
        kmeans.fit(data)
        w.append(kmeans.inertia_)
    plt.plot(range(1, 11), w)
    plt.title("Elbow Plot")
    plt.xlabel("n_clusters")
    plt.ylabel("WCSS score")
    plt.show()

def plot_cluster(data: pd.DataFrame, label_idx: str = None, title:str = 'Cluster'):
    x_idx, y_idx = data.columns[0], data.columns[1]
    plt.xlabel(x_idx)
    plt.ylabel(y_idx)
    plt.title(title)

    if label_idx is None:
        plt.scatter(data[x_idx], data[y_idx], s=50, alpha=.6)
        plt.show()
        return

    labels = data[label_idx].unique()
    colors = ['purple', 'green', 'blue']
    for i in labels:
        df = data.loc[data[label_idx] == i]
        plt.scatter(df[x_idx], df[y_idx], s=50, alpha=.6, c=colors[i])

    plt.show()

def show_table(data: pd.DataFrame, label:str):
    data['Country'] = data.index
    data= data[['Country',label]]
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=data.values, colLabels=data.columns, loc='center')
    fig.tight_layout()
    plt.show()

def use_clustering(data: pd.DataFrame, x_idx: str, y_idx: str):
    year = '2019'
    df = data.loc[[x_idx, y_idx], [year]]
    df = df.unstack(level=0)
    df = df.dropna()
    df.columns = df.columns.droplevel()
    
    # Visualize before clustering
    plot_cluster(df, title=f'Scatterplot of countries [{year}] before clustering')

    # For optimum number of n_cluster
    kmean_elbow_plot(df)
    n_clusters = 3

    # Run k mean clustering and plot
    kmeans = KMeans(n_clusters)
    lbl = 'Label'
    df[lbl] = kmeans.fit_predict(df)
    plot_cluster(df, label_idx=lbl, title=f'Scatterplot of clusters of countries[{year}]')

    # Show tables
    for i in range(n_clusters):
        show_table(df.loc[df[lbl]==i], label='Label')




def main():
    raw_data = pd.read_excel('wb_data.xlsx', index_col=[0, 1])

    # Forward fill 10 then backward fill 5 consecutive NaN values. Drop rest of the NaN containing rows
    # otherwise they will bias our analysis
    clean_data = raw_data.ffill(axis='columns', limit=10).bfill(
        axis='columns', limit=5).dropna(axis='index')

    # Heatmap for choosing appropriate indices for clustering
    corr_heat_map(clean_data)

    # For getting better insight making scatter matrix
    scatter_matrix(clean_data)

    # Run clustering on chosen values
    x_idx = 'Poverty headcount ratio at $3.65 a day (2017 PPP) (% of population)'
    y_idx = 'Access to electricity (% of population)'
    use_clustering(clean_data, x_idx, y_idx)


if __name__ == "__main__":
    main()
