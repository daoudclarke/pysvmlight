# Bismillahi-r-Rahmani-r-Rahim

import unittest

from svmlight import SupportVector

class test_SupportVectorTestCase(unittest.TestCase):
    def testConstruction(self):
        self.assertEqual(str(SupportVector([1,2,3])),
                         "SupportVector([1,2,3])")

    def testNoZerosAllowed(self):
        self.assertRaises(ValueError, SupportVector, [0,1,2])

    def testInputIsOrdered(self):
        self.assertEqual(str(SupportVector([3,1,2])),
                         "SupportVector([1,2,3])")

    def testEmptyInput(self):
        self.assertEqual(str(SupportVector([])),
                         "SupportVector([])")
    
if __name__ == '__main__':
    unittest.main()
