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
        ('th'): 'grey',
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
        """Returns the dataframe"""
        return self.data


    def get_scaled_data(self, scaler='maxabs', data=None):
        """Returns a normalised version of the dataframe.
        The built-in scalers are:
        * 'maxabs' for sklearn.preprocessing.MaxAbsScaler
        * 'standard' for sklearn.preprocessing.StandardScaler
        """
        scalers = {'maxabs': MaxAbsScaler, 'standard': StandardScaler}
        if scaler in scalers:
            scaler = scalers[scaler]
        data = data if data is not None else self.data
        # Standardize the data to have a mean of ~0 and a variance of 1
        scaled_df = pd.DataFrame(scaler().fit_transform(data))
        scaled_df.columns = data.columns
        scaled_df.index = data.index
        return scaled_df


    def mark_expandable_cols(self, expandable_cols, limit=3):
        """Concatenates all dataframes into one and returns a dictionary in which the keys
        are the original columns names and the values are either an empty list or a list
        containing that column's X most frequent values. The columns to be marked like this
        are taken as a parameter.
        """
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
        """Returns a vector for the input dataframe.
        * self.expanded_columns tells which columns should be replaced, and by which of their
        values. The frequency of these values is calculated and added to the vector as a feature
        * in the case of numerical columns, their mean value is taken
        * all other columns are converted to bool and the frequency of True values taken
        """
        # there probably is a much better way to do this...
        vector = []
        for col_name, col_type in dataframe.dtypes.items():
            new_columns = self.expanded_columns.get(col_name)
            if len(new_columns) > 0:  #  "if new_columns" does not work
                # Counts all the values in this column and normalises them (to percertages)
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
        """Returns a feature matrix containing a vector for each dataframe
        using self.create_vector().
        """
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
        """Creates a dendrogram for the input data.
        The default data is normalised.
        """
        data = data if data is not None else self.scaled_data
        plt.figure(figsize=(20, 10))
        sch.dendrogram(
            sch.linkage(data, method=method),
            labels=data.index,
            leaf_rotation=90
        )
        plt.show()


    def show_clustering(self, data, algorithm, n_clusters):
        """Calculates and prints clusterings for the input data
        using the specified algorithm.
        """
        data = data if data is not None else self.scaled_data
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
        """Calculates and prints clusterings using self.show_clustering().
        The algorithms in use are sklearn.cluster.KMeans and
        sklearn.cluster.AgglomerativeClustering. The default data is
        the normalised feature matrix.
        """
        data = data if data is not None else self.scaled_data
        #  data = data or self.scaled_data
        for clustering in (KMeans, AgglomerativeClustering):
            self.show_clustering(data, clustering, n_clusters)


    # PCA analysis (variance, 2d plot, inertia): code adapted from Dmitriy Kavyazin's post at
    # https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2

    def find_pca_components(self, data=None, n_components=15):
        """Calculates PCA components for the input data.
        The default data is normalised.
        Returns a tuple with the PCA and its componens
        """
        data = data if data is not None else self.scaled_data
        #  X_std = StandardScaler().fit_transform(data)
        # Create a PCA instance: pca
        pca = PCA(n_components)
        principalComponents = pca.fit_transform(data)
        # Save components to a DataFrame
        pca_components = pd.DataFrame(principalComponents)
        return pca, pca_components


    def draw_pca_analysis(self):
        """Draws all available graphs using default values"""
        pca, pca_components = self.find_pca_components()
        self.draw_pca_variance(pca, pca_components)
        self.draw_2d_scatterplot(
            pca_components.iloc[:, :2],
            labels=('PCA 1', 'PCA 2'),
        )
        self.draw_clusters_inertia_graph(title='KMeans clustering vs inertia')
        #  self.draw_clusters_inertia_graph(
        #      pca_components.iloc[:, :3], title='3 PCA components vs inertia'
        #  )
        self.draw_3d_scatterplot(pca_components.iloc[:, :3], labels=self.labels)


    def draw_pca_variance(self, pca, pca_components):
        """Draws a PCA vs variance bar graph"""
        features = range(pca.n_components_)
        plt.figure(figsize=(20, 10))
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.show()


    def draw_2d_scatterplot(self, data, labels):
        """Draws a 2d scatterplot.
        The input data are a two-column dataframe and a a list of 2 labels.
        """
        x_list, y_list = data.iloc[:, 0], data.iloc[:, 1]
        x_label, y_label = labels
        fig, ax = plt.subplots(figsize=(20, 14))
        # The following does not work:
        #  colours = self.assign_colours()
        ax.scatter(x_list, y_list, alpha=.1, color='black')
        for i, label in enumerate(self.labels):
            ax.annotate(label, (x_list[i], y_list[i]))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


    def draw_clusters_inertia_graph(self, data=None, title=None):
        """Draws a "number of clusters vs inertia" graph.
        The default input data is the normalised feature vector.
        """
        data = data if data is not None else self.scaled_data
        ks = range(1, 10)
        inertias = []
        for k in ks:
            model = KMeans(n_clusters=k).fit(data)
            inertias.append(model.inertia_)
        plt.figure(figsize=(15, 10))
        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        if title:
            plt.title(title)
        plt.xticks(ks)
        plt.show()


    def assign_colours(self, data=None):
        """Returns a list containing the corresponding colour for each language.
        The default input is the normalised feature matrix.
        """
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


    def draw_3d_scatterplot(self, dataframe, labels, filename='clusters_color.html'):
        """Saves and shows a 3d scatterplot. Expects a 3-part dataframe as input
        and a list of labels.
        """
        assert len(dataframe.columns), 3
        x_list, y_list, z_list = (dataframe[col] for col in dataframe.columns)
        d3_labels = []
        for n, label in enumerate(labels):
            d3_labels.append({'text': label, 'x': x_list[n], 'y': y_list[n], 'z': z_list[n]})
        colours = self.assign_colours()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=x_list,
                y=y_list,
                z=z_list,
                mode='markers',
                marker={'color': colours}
            )
        )
        fig.update_layout(scene=go.layout.Scene(annotations=d3_labels))
        if filename:
            fig.write_html(filename, auto_open=True)
        else:
            fig.show()
