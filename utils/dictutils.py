from exceptions import IndexError, KeyError

__author__ = 'makov'

def get_value(dic, keys_chain, key):
    """
    Return value of key (up toward the tree)
    Searching the key

    :param dic: dictionary, where to search
    :param keys_chain: list of keys. For dic['a']['b'] should be ['a','b']
    :param key: key of searching value
    :return:
    :raise: KeyError
    """
    keys_chain = list(keys_chain)

    while True:
        try:
            tmp_node = get_node(dic,keys_chain)
            if key in tmp_node:
                return tmp_node[key]
            else:
                keys_chain.pop(-1)
        except IndexError:
            raise KeyError


def get_node(dic,keys_chain):
    """
    Return node from dictionary by  chain of keys

    :param keys_chain: list of keys
    :raise : KeyError
    """
    node = dict(dic)
    for key in keys_chain:
        if key in node:
            node = node[key]
        else:
            raise KeyError
    return node


def try_get_value(dic, keys_chain, key,default_value=None):
    """
    Safe version of get_value, if key not exist return def_value
    :param dic:
    :param keys_chain:
    :param key:
    :param default_value:
    :return:
    """
    try:
        return get_value(dic, keys_chain, key)
    except KeyError:
        return default_value