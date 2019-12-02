import os
import re
from collections import defaultdict

import pandas as pd

import pyconll


WANTED_POS = ("NOUN", "PROPN", "PRON")
EXTERNAL_CASE = {
        'zh': ('的', 'PART', False),
        'ja': ('の', 'ADP', True),
        }


def check_sufficent_tags(data):
    for sent in data:
        for token in sent:
            if token.deprel in ("nmod:poss", "nmod:own", "nmod:att"):
                return token.deprel
    return


def list_external_cases(sentence, lang, sufficient_tag):
    # gather external case markings
    external_cases = defaultdict(set)
    for token in sentence:
        try:
            head = sentence[token.head]
        except (KeyError, ValueError):
            continue
        if (
            token.deprel != 'case'
            or head.deprel not in (sufficient_tag, 'nmod')
            or head.upos not in WANTED_POS
        ):
            continue
        if lang in EXTERNAL_CASE:
            form, upos, immediate = EXTERNAL_CASE.get(lang)
            if not (
                    token.form == form
                    and token.upos == upos
                    #  and (len(xpos) == 0 or token.xpos in xpos)
                    ):
                continue
            if immediate and int(head.id) != int(token.id) - 1:
                continue

        external_cases[head].add(token)

    return external_cases


def extract_features(filename, lang):
    def find_tags(token, *tags):
        tagset = set()
        for tag in tags:
            if tag in token.feats:
                tagset |= token.feats.get(tag)
        return '+'.join(sorted(tagset))


    features = []
    train = pyconll.load_from_file(filename)
    sufficient_tag = check_sufficent_tags(train)

    for sentence in train:
        external_cases = list_external_cases(sentence, lang, sufficient_tag)
        for token in sentence:
            if lang in EXTERNAL_CASE and token not in external_cases:
                continue
            if token.upos not in WANTED_POS:
                continue
            # either the deprel tag is sufficient, or:
            # (1) the deprel is nmod and the case is genitive; or
            # (2) the deprel is nmod and the case is marked externally
            if not (
                (token.deprel == sufficient_tag)
                or (
                    token.deprel == "nmod"
                    and (
                        "Gen" in token.feats.get("Case", {})
                        or external_cases.get(token)
                    )
                )
            ):
                continue
            head = sentence[token.head]
            if sentence[token.head].upos not in WANTED_POS:
                continue
            distance = int(token.head) - int(token.id)
            head_marked = ', '.join(sorted(p for p in head.feats.get("Person[psor]", {})))
            head_case = ', '.join(t for t in head.feats.get("Case", {}))
            dep_marked = "Gen" in token.feats.get("Case", {})
            dep_case = ', '.join(t for t in token.feats.get("Case", {}))
            if token in external_cases:
                external_case = ', '.join((e.form for e in external_cases.get(token)))
            else:
                external_case = ''
            head_tags = find_tags(head, "Number", "Number[psor]", "Person", "Person[psor]")
            dep_tags = find_tags(token, "Number", "Number[psed]", "Person", "Person[psed]")
            psor_in_dep = find_tags(token, "Number[psor]", "Person[psor]")
            feats = ', '.join(
                              t for t in token.feats.keys()
                              if t not in (
                                  "Case",
                                  "Number",
                                  "Person",
                                  "Number[psed]",
                                  "Person[psed]",
                                  "Number[psor]",
                                  "Person[psor]",
                              )
                    )

            sentence_features = [
                head.upos,
                head.deprel,
                head_marked,
                head_case,
                token.upos,
                #  token.deprel,
                dep_marked,
                dep_case,
                distance,
                distance < 0,
                external_case,
                psor_in_dep,
                head_tags,
                dep_tags,
                feats,
            ]

            features.append(sentence_features)

    return features


def handle_languages(data_path):
    all_languages = dict()
    for dirpath, _, filenames in os.walk(data_path):
        for filename in sorted(filenames):
            print(f"Processing {filename}")
            lang_file = os.path.join(dirpath, filename)
            lang, treebank = filename.split(".")[0].split("_")
            features = extract_features(lang_file, lang)
            all_languages[f"{lang}_{treebank}"] = pd.DataFrame(
                features,
                columns=[
                    "hd_pos",
                    "hd_deprel",
                    "hd_marked",
                    "hd_case",
                    "dp_pos",
                    #  "dep_deprel",
                    "dp_marked",
                    "dp_case",
                    "dist",
                    "head_1st",
                    "ext_case",
                    "dp_psor",
                    "hd_tags",
                    "dp_tags",
                    "feats",
                ],
            )

    return all_languages
