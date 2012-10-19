__author__ = 'makov'
import tomofrontend.configutils.yamlutils as yamlutils

from utils.mypprint import pprint


def test_yaml_reader():
    test_yaml_filename = '/media/WD_ext4/bones/ikran/2012_02_02/F1-M2/exp_description.yaml'
    res = yamlutils.read_yaml(test_yaml_filename)
    pprint(list(res))

if __name__ == "__main__":

    test_yaml_reader()
