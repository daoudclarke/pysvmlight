# Bismillahi-r-Rahmani-r-Rahim

import unittest

from svmlight import SupportVector

class SupportVectorTestCase(unittest.TestCase):
    def testConstruction(self):
        self.assertEqual(str(SupportVector([1,2,3])),
                         "SupportVector([1, 2, 3])")

    def testNoZerosAllowed(self):
        self.assertRaises(ValueError, SupportVector, [0,1,2])

    def testInputIsOrdered(self):
        self.assertEqual(str(SupportVector([3,1,2])),
                         "SupportVector([1, 2, 3])")

    def testEmptyInput(self):
        self.assertEqual(str(SupportVector([])),
                         "SupportVector([])")

    def testIter(self):
        s = SupportVector([1,2,3])
        vals = [x for x in s]
        self.assertEqual(str(vals), "[1, 2, 3]")

    def testNoneVector(self):
        self.assertEqual(str(SupportVector(None)),
                         "SupportVector(None)")

    def testNoneVectorLengthError(self):
        s = SupportVector(None)
        self.assertRaises(ValueError, len, s)

    def testNoneVectorListError(self):
        s = SupportVector(None)
        self.assertRaises(ValueError, list, s)
    
if __name__ == '__main__':
    unittest.main()
