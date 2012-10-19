"""
Test natural sorting.
"""
__author__ = 'Alexey Buzmakov'

import unittest
import utils.natsort as natsort

input_list = ['0', '02', '1', '10', '11', 'z', 'a', '0a', '0z', '01']
ref_list = ['0', '0a', '0z', '1', '01', '02', '10', '11', 'a', 'z']


class NatsortTestCase(unittest.TestCase):
    def test_natsort(self):
        self.assertEqual(natsort.natsorted(input_list), ref_list)

if __name__ == '__main__':
    unittest.main()
