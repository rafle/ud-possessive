from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


SPLITTABLE_COLS = ('hd_pos', 'hd_deprel', 'hd_case', 'dp_pos')

class DataAnalyser:
    def __init__(self, dataframe_collection):
        self.dataframe = dataframe_collection
        self.data = self.handle_data(self.dataframe, SPLITTABLE_COLS)


    def mark_splittable_columns(self, dataframe, splittable_cols, limit=3):
        concat_df = pd.concat(dataframe.values())
        columns = OrderedDict()
        for col_name in concat_df.keys():
            if col_name in splittable_cols:
                columns[col_name] = concat_df[col_name].value_counts()[:limit].keys()
            else:
                columns[col_name] = []
        return columns


    def create_vector(self, lang, dataframe, splittable):
        # there probably is a much better way to do this...
        vector = []
        for col_name, col_type in dataframe.dtypes.items():
            new_columns = splittable.get(col_name)
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


    def handle_data(self, dataframe_collection, splittable_columns=SPLITTABLE_COLS):
        col_names = self.mark_splittable_columns(self.dataframe, splittable_columns)
        series = [self.create_vector(lang, df, col_names) for lang, df in dataframe_collection.items()]
        new_col_names = []
        for col_name, subcats in col_names.items():
            if len(subcats) == 0:
                new_col_names.append(col_name)
                continue
            for subcat in subcats:
                new_col_names.append(f"{col_name}_{subcat}")
        vector_df = pd.DataFrame(series)
        vector_df.columns = new_col_names
        return vector_df


    def draw_dendrogram(self):
        dendrogram = sch.dendrogram(sch.linkage(self.data, method='ward'))

