#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World bank data clustering
@author: Muhammad Arafat Azam (id: 21087019)
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from textwrap import wrap

YEARS = [f'{i}' for i in range(1990, 2021, 5)]


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


def main():
    raw_data = pd.read_excel('wb_data.xlsx', index_col=[0, 1])

    # Forward fill then backward fill consecutive 3 NaN values. Drop rest of the NaN containingrows
    # otherwise they will bias our analysis
    clean_data = raw_data.ffill(axis='columns', limit=3).bfill(
        axis='columns', limit=5).dropna(axis='index')

    # Heatmap for choosing appropriate indices for clustering
    corr_heat_map(clean_data)

    # For getting better insight making scatter matrix
    scatter_matrix(clean_data)


if __name__ == "__main__":
    main()
