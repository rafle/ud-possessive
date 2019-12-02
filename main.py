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
        }


prepare_data.main(SOURCE_DIR, DATA_DIR, WANTED)
langs = features_extractor.handle_languages(DATA_DIR)

#  for lang, df in langs.items():
#      print(lang)
#      print(df)
    #  print(df.describe())


analysis = DataAnalyser(langs)
analysis.draw_dendrogram()
