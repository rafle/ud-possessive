import prepare_data
import features_extractor
from data_analyser import DataAnalyser


SOURCE_DIR = 'source'
DATA_DIR = 'data'
TEST_DIR = 'test'

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

EXPANDABLE_COLS = ('hd_pos', 'hd_deprel', 'hd_case', 'dp_pos')

prepare_data.main(SOURCE_DIR, DATA_DIR, WANTED)
langs = features_extractor.handle_languages(DATA_DIR)

#  for lang, df in langs.items():
#      print(lang)
#      print(df)
    #  print(df.describe())


analysis = DataAnalyser(langs, EXPANDABLE_COLS)
analysis.draw_dendrogram()
analysis.draw_pca()
analysis.show_all_clusterings()
