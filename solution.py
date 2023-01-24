#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World bank data analysis
@author: Muhammad Arafat Azam (id: 21087019)
"""

import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from textwrap import wrap

YEARS = ['1991', '1995', '1999', '2003', '2007', '2011', '2015', '2019']
CO2_IDX = 'CO2 emissions (metric tons per capita)'


def scatter_matrix(data: pd.DataFrame):
    #Filter by selected years for faster result
    df = data[YEARS]

    #Unstacking 2nd index (Countries) into columns and then transposing
    df = df.unstack(level=1).T

    axs = pd.plotting.scatter_matrix(df, figsize=(7, 7), s=20, alpha=0.8)
    for ax in axs.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    plt.suptitle('Scatter Matrix')
    plt.tight_layout()
    plt.show()


def corr_heat_map(data: pd.DataFrame):
    #Unstacking 2nd index (Countries) into columns and then transposing
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


def plot_before_clustering(data: pd.DataFrame, idx: str):
    df = data.loc[[idx, CO2_IDX], YEARS]
    df = df.fillna(0.0)
    col = 3
    rows = ceil(len(YEARS)/3)
    iter = 1
    for year in YEARS:
        x = df.xs(CO2_IDX, level=0)[[year]]
        y = df.xs(idx, level=0)[[year]]
        plt.scatter(x,y,s=5)
        plt.title(f'{idx} [{year}]')
        plt.xlabel(CO2_IDX)
        plt.ylabel(idx)
        plt.show()


def main():
    raw_data = pd.read_excel('wb_data.xlsx', index_col=[0, 1])

    # Forward fill then backward fill consecutive 3 NaN values. Drop rest of the NaN containingrows 
    # otherwise they will bias our analysis
    clean_data = raw_data.ffill(axis='columns', limit=3).bfill(axis='columns', limit=5).dropna(axis='index')

    # Heatmap for choosing appropriate indices for clustering
    corr_heat_map(clean_data)

    # For getting better insight making scatter matrix
    scatter_matrix(clean_data)


if __name__ == "__main__":
    main()
