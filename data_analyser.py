from collections import OrderedDict
from itertools import zip_longest

import pandas as pd
import matplotlib.pyplot as plt

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import plotly.graph_objects as go


class DataAnalyser:
    LANG_COLOURS = {
        ('ru'): 'orange',
        ('et', 'fi', 'hu', 'sme', 'kpv', 'myv', 'krl', 'mdf'): 'red',
        ('tr', 'ug', 'kk'): 'green',
        ('bxr'): 'violet',
        ('zh'): 'paleturquoise',
        ('th'): 'brown',
        ('ko'): 'olive',
        ('ja'): 'blue',
    }


    def __init__(self, dataframe_collection, expandable_cols):
        self.df_collection = dataframe_collection
        self.expanded_columns = self.mark_expandable_cols(expandable_cols)
        self.data = self.simplify_df()
        self.scaled_data = self.get_scaled_data()
        self.labels = self.data.index


    def get_data(self):
        return self.data


    def get_scaled_data(self, scaler=MaxAbsScaler, data=None):
        data = data if data is not None else self.data
        # Standardize the data to have a mean of ~0 and a variance of 1
        scaled_df = pd.DataFrame(scaler().fit_transform(data))
        scaled_df.columns = data.columns
        scaled_df.index = data.index
        return scaled_df


    def mark_expandable_cols(self, expandable_cols, limit=3):
        expandable_cols = expandable_cols
        concat_df = pd.concat(self.df_collection.values())
        columns = OrderedDict()
        for col_name in concat_df.keys():
            if col_name in expandable_cols:
                columns[col_name] = concat_df[col_name].value_counts()[:limit].keys()
            else:
                columns[col_name] = []
        return columns


    def create_vector(self, lang, dataframe):
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


    def simplify_df(self):
        series = [
            self.create_vector(lang, df)
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


    def draw_dendrogram(self, data=None, method='ward'):
        data = data if data is not None else self.scaled_data
        plt.figure(figsize=(20, 10))
        sch.dendrogram(
            sch.linkage(data, method=method),
            labels=data.index,
            leaf_rotation=90
        )
        plt.show()


    def show_clustering(self, data, algorithm, n_clusters):
        clusters = algorithm(n_clusters=n_clusters).fit(data)
        labels = clusters.labels_
        print(f"\nClustering with {algorithm.__name__}:\n")
        clusters = [[] for r in range(n_clusters)]
        for group, lang in zip(labels, data.index):
            clusters[group].append(lang)
        row_format = '{:<15}' * len(clusters)
        cluster_names = [f"Cluster {n}" for n, _ in enumerate(clusters)]
        print(row_format.format(*cluster_names))
        print('-' * 15 * len(clusters))
        for c in zip_longest(*clusters, fillvalue=''):
            print(row_format.format(*c))


    def show_all_clusterings(self, data=None, n_clusters=5):
        data = data if data is not None else self.scaled_data
        #  data = data or self.scaled_data
        for clustering in (KMeans, AgglomerativeClustering):
            self.show_clustering(data, clustering, n_clusters)


    # PCA analysis (variance, 2d plot, inertia): code adapted from Dmitriy Kavyazin's post at
    # https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2

    def find_pca_components(self, data=None):
        data = data if data is not None else self.scaled_data
        #  X_std = StandardScaler().fit_transform(data)
        # Create a PCA instance: pca
        pca = PCA(n_components=15)
        principalComponents = pca.fit_transform(data)
        # Save components to a DataFrame
        pca_components = pd.DataFrame(principalComponents)
        return pca, pca_components


    def draw_pca_analysis(self):
        pca, pca_components = self.find_pca_components()
        self.draw_pca_variance(pca, pca_components)
        self.draw_pca_2d_scatterplot(pca_components)
        self.draw_pca_inertia(pca_components)
        self.draw_pca_3d_scatterplot(pca_components)


    def draw_pca_variance(self, pca, pca_components):
        features = range(pca.n_components_)
        plt.figure(figsize=(20, 10))
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.show()


    def draw_pca_2d_scatterplot(self, pca_components):
        fig, ax = plt.subplots(figsize=(20, 14))
        #  colours = self.assign_colours()
        ax.scatter(pca_components[0], pca_components[1], alpha=.1, color='black')
        for i, label in enumerate(self.labels):
            ax.annotate(label, (pca_components[0][i], pca_components[1][i]))
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.show()


    def draw_pca_inertia(self, pca_components):
        #  pca3 = pca_components.iloc[:, :3]
        ks = range(1, 10)
        inertias = []
        for k in ks:
            # Create a KMeans instance with k clusters: model
            model = KMeans(n_clusters=k)
            # Fit model to samples
            model.fit(pca_components.iloc[:, :3])
            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)
        plt.figure(figsize=(15,10))
        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(ks)
        plt.show()


    def assign_colours(self, data=None):
        data = data if data is not None else self.scaled_data
        colours = []
        for label in data.index:
            # I cannot get for/else to work...
            found = False
            lang = label.split('_')[0]
            for lang_group, colour in DataAnalyser.LANG_COLOURS.items():
                if lang in lang_group:
                    colours.append(colour)
                    found = True
            if not found:
                colours.append('black')
        return colours


    def draw_pca_3d_scatterplot(self, pca_components, filename='clusters_color.html'):
        d3_labels = []
        for n, label in enumerate(self.labels):
            x, y, z = pca_components.iloc[n, :3]
            d3_labels.append({'text': label, 'x': x, 'y': y, 'z': z})
        colours = self.assign_colours()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=pca_components[0],
                y=pca_components[1],
                z=pca_components[2],
                mode='markers',
                marker={'color': colours}
            )
        )
        fig.update_layout(scene=go.layout.Scene(annotations=d3_labels))
        if filename:
            fig.write_html(filename, auto_open=True)
        else:
            fig.show()
