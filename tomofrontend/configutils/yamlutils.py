__author__ = 'makov'
import yaml


def read_yaml(yaml_filename):
    docs = []
    with open(yaml_filename) as f:
        docs = yaml.load_all(f)
        docs = list(docs)
#        docs=yaml.dump_all(ya, encoding='utf8', allow_unicode=True)
    return docs


def save_yaml(docs, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(docs, stream=f, encoding='utf8', allow_unicode=True)
