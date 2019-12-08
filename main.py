import pickle

import prepare_data
import features_extractor
from data_analyser import DataAnalyser


SOURCE_DIR = 'source'
DATA_DIR = 'data'
TEST_DIR = 'test'
LANGS_FILE = 'langs.db'
EXPANDABLE_COLS = ('hd_pos', 'hd_deprel', 'hd_case', 'dp_pos')
WANTED = {
        'Buryat': {},
        'Kazakh': {},
        'Uyghur': {},
        'Erzya': {},
        'Karelian': {},
        'Korean': {'only': ('PUD')},
        'Russian': {},
        'Estonian': {},
        'Finnish': {},
        'Turkish': {},
        'Komi_Zyrian': {},
        'Chinese': {'except': ('GSD', 'GSDSimp')},
        'Moksha': {},
        'North_Sami': {},
        'Hungarian': {},
        'Japanese': {'only': ('GSD', 'PUD', 'Modern')},
        'Thai': {},
        }


try:
    with open(LANGS_FILE, 'rb') as f:
        langs = pickle.load(f)
except FileNotFoundError:
    prepare_data.main(SOURCE_DIR, DATA_DIR, WANTED)
    langs = features_extractor.handle_languages(DATA_DIR)
    with open(LANGS_FILE, 'wb') as f:
        pickle.dump(langs, f, pickle.HIGHEST_PROTOCOL)

dal = DataAnalyser(langs, EXPANDABLE_COLS)
dal.show_all_clusterings(n_clusters=4)
dal.show_all_clusterings(n_clusters=5)
dal.show_all_clusterings(n_clusters=6)
dal.draw_dendrogram()
dal.draw_pca_analysis()
