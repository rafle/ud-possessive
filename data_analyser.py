from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA


#  EXPANDABLE_COLS = ('hd_pos', 'hd_deprel', 'hd_case', 'dp_pos')

class DataAnalyser:
    def __init__(self, dataframe_collection, expandable_cols):
        self.df_collection = dataframe_collection
        self.data = self.simplify_df():  #  self.df_collection, EXPANDABLE_COLS)
        self.expanded_columns = self.mark_expandable_cols(expandable_cols)


    def get_data(self):
        return self.data


    def mark_expandable_cols(self, expandable_cols, limit=3):
        expandable_cols = expandable_cols or self.expanded_columns
        concat_df = pd.concat(self.df_collection.values())
        columns = OrderedDict()
        for col_name in concat_df.keys():
            if col_name in expandable_cols:
                columns[col_name] = concat_df[col_name].value_counts()[:limit].keys()
            else:
                columns[col_name] = []
        return columns


    def create_vector(self, lang, dataframe):  # , expandable):
        # there probably is a much better way to do this...
        vector = []
        for col_name, col_type in dataframe.dtypes.items():
            new_columns = self.expanded_columns.get(col_name)
            if len(new_columns) > 0:  #  "if new_columns" does not work
                counts = dataframe[col_name].value_counts(normalize=True)
                for new_column in new_columns:
                    vector.append(counts.get(new_column, 0))
            elif col_type == 'int64':
                vector.append(dataframe[col_name].mean())
            else:
                bool_counts = dataframe[col_name].astype(bool).value_counts(normalize=True)
                vector.append(bool_counts.get(True, 0))
        return pd.Series(vector, name=lang)


    def simplify_df(self):  #  , dataframe_collection):  # , splittable_columns=EXPANDABLE_COLS):
        #  col_names = self.mark_expandable_cols(self.df_collection, splittable_columns)
        series = [
                    self.create_vector(lang, df, self.expanded_columns)
                    for lang, df in self.df_collection.items()
                 ]
        new_col_names = []
        for col_name, subcats in self.expanded_columns.items():
            if len(subcats) == 0:
                new_col_names.append(col_name)
                continue
            for subcat in subcats:
                new_col_names.append(f"{col_name}_{subcat}")
        vector_df = pd.DataFrame(series)
        vector_df.columns = new_col_names
        return vector_df


    def draw_dendrogram(self, method='ward'):
        sch.dendrogram(
                        sch.linkage(self.data, method=method),
                        labels=self.data.index,
                        leaf_rotation=90
                )


    def show_clustering(self, algorithm, n_clusters):
        clusters = algorithm(n_clusters=n_clusters).fit(self.data)
        labels = clusters.labels_
        #  self.show_clusters(labels)
        print(f"Clustering with {algorithm.__name__}:\n")
        for cluster_n in range(max(labels) + 1):
            print(f"Cluster {cluster_n}:")
            for group, lang in zip(labels, self.data.index):
                if group == cluster_n:
                    print(f"  {lang}")
            print()
        print("\n\n")


    def draw_pca(self):
        print('Dimensionality reduction via PCA decomposition:\n')
        X_pca = PCA(2).fit_transform(self.data)
        x = X_pca[:,0]
        y = X_pca[:,1]

        fig, ax = plt.subplots(figsize=(20, 14))
        ax.scatter(x, y)
        for i, label in enumerate(self.data.index):
             ax.annotate(label, (x[i], y[i]))


    def show_all_clusterings(self, n_clusters=5):
        for clustering in (KMeans, AgglomerativeClustering):
            self.show_clustering(clustering, n_clusters)
