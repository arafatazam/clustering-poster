#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World bank data analysis
@author: Muhammad Arafat Azam (id: 21087019)
"""

import pandas as pd
import matplotlib.pyplot as plt


def scatter_matrix(data: pd.DataFrame, country: str):
    df = data.xs(country, level=1)
    axs = pd.plotting.scatter_matrix(df.T, figsize=(9, 9), s=20, alpha=0.8)
    for ax in axs.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    plt.suptitle(f'Scatter Matrix: {country}')
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_excel('wb_data.xlsx', index_col=[0, 1])
    print(df.index.get_level_values(0).unique())

    # In order to find out right candidate for the clustering viewing scatter plot of several countries
    countries = ['United States', 'United Kingdom',
                 'China', 'India', 'Bangladesh', 'Nigeria']
    for country in countries:
        scatter_matrix(df, country)


if __name__ == "__main__":
    main()
