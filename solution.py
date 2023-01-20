#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World bank data analysis
@author: Muhammad Arafat Azam (id: 21087019)
"""

import pandas as pd
import matplotlib.pyplot as plt


def corr_heat_map(data: pd.DataFrame, country: str):
    df = data.xs(country, level=1)
    df = df.dropna(axis='columns')
    print(df.columns)
    df = df.T
    corr = df.corr()
    print(corr.columns, corr.index)
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(corr, interpolation='nearest')
    fig.colorbar(im, orientation='vertical', fraction=0.05)
    ax.set_xticklabels(['']+corr.columns.to_list(), rotation=90, fontsize=6)
    ax.set_yticklabels(['']+corr.index.to_list(), rotation=0, fontsize=6)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, round(corr.to_numpy()[i, j], 2),
                           ha="center", va="center", color="black")
    plt.show()


def main():
    df = pd.read_excel('wb_data.xlsx', index_col=[0, 1])
    # greenhouse_gas_emission_barchart(df)
    corr_heat_map(df, "United States")


if __name__ == "__main__":
    main()
