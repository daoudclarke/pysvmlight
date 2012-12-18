# Bismillahi-r-Rahmani-r-Rahim

import unittest

from svmlight import Model

class ModelTestCase(unittest.TestCase):
    def testConstructionCreatesInvalidModel(self):
        m = Model()
        self.assertRaises(ValueError, m.__getattribute__, 'bias')
    
if __name__ == '__main__':
    unittest.main()
