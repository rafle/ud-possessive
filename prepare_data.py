import os
from collections import defaultdict

import requests

ORG = 'UniversalDependencies'
LINK = f"https://api.github.com/orgs/{ORG}/repos?per_page=100"


def fetch_repos(link=LINK, repos={}, wanted_langs=None):
    repos_page = requests.get(link)
    for repo in repos_page.json():
        if not repo['name'].startswith('UD'):
            continue
        lang, treebank = repo['name'].replace('UD_', '').split('-')
        if wanted_langs:
            if lang not in wanted_langs or treebank in wanted_langs[lang].get('except', {}):
                continue
            only_wanted = wanted_langs[lang].get('only', ())
            if only_wanted and treebank not in only_wanted:
                continue
        repos[repo['name']] = repo['url']
    try:
        next_page = repos_page.links['next']['url']
        fetch_repos(next_page, repos, wanted_langs)
    except KeyError:
        pass
    return repos


def fetch_conllus(repos, folder):
    os.makedirs(folder, exist_ok=True)
    print('Downloading files')
    for repo_url in repos.values():
        docs = requests.get(repo_url + '/contents')
        for doc in docs.json():
            try:
                if not doc['name'].endswith('conllu'):
                    continue
            except TypeError:
                continue
            filename = os.path.join(folder, doc['name'])
            if os.path.isfile(filename):
                continue
            print(f"Downloading {doc['name']}")
            doc_text = requests.get(doc['download_url']).text
            with open(filename, 'w') as f:
                f.write(doc_text)
    print('Done\n')


def group_files(source_dir):
    langtree_dict = defaultdict(list)
    for _, _, filenames in os.walk(source_dir):
        for filename in filenames:
            langtree = filename.split('-', 1)[0]
            if not langtree:
                continue
            langtree_dict[langtree].append(filename)
    return langtree_dict


def concatenate_files(filename, source_files, source_dir, data_dir):
    data_file = os.path.join(data_dir, filename)
    if os.path.isfile(data_file):
        return
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, filename + '.conllu')
    with open(filename, 'w') as outfile:
        for source_file in source_files:
            with open(os.path.join(source_dir, source_file)) as infile:
                for line in infile:
                    outfile.write(line)


def main(source_dir, data_dir, wanted_langs_dic):
    if not os.path.isdir(data_dir) or len(os.listdir(data_dir)) == 0:
        fetch_conllus(fetch_repos(wanted_langs=wanted_langs_dic), source_dir)
    print('Preparing files')
    langtree_dict = group_files(source_dir)
    for filename, source_files in langtree_dict.items():
        concatenate_files(filename, source_files, source_dir, data_dir)
    print('Done\n')

