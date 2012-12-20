# Bismillahi-r-Rahmani-r-Rahim

import unittest

from svmlight import SupportVector

class SupportVectorTestCase(unittest.TestCase):
    def testConstruction(self):
        self.assertEqual(str(SupportVector([(1,1.),(2,2.),(3,3.)])),
                         "SupportVector({1: 1.0, 2: 2.0, 3: 3.0})")

    def testNoZerosAllowed(self):
        self.assertRaises(ValueError, SupportVector, [(0,1.), (1,1.), (2,2.)])

    def testInputIsOrdered(self):
        l = [(3, 1.0), (1, 1.0), (2, 2.0)]
        self.assertEqual(str(SupportVector(l)),
                         "SupportVector(%s)" % dict(l).__repr__())

    def testEmptyInput(self):
        self.assertEqual(str(SupportVector([])),
                         "SupportVector({})")

    def testIter(self):
        l = [(1, 1.0), (2, 2.0), (5, 1.0)]
        s = SupportVector(l)
        vals = [x for x in s]
        self.assertEqual(str(vals), str(l))

    def testNoneVector(self):
        self.assertEqual(str(SupportVector(None)),
                         "SupportVector(None)")

    def testNoneVectorLengthError(self):
        s = SupportVector(None)
        self.assertRaises(ValueError, len, s)

    def testNoneVectorListError(self):
        s = SupportVector(None)
        self.assertRaises(ValueError, list, s)

    def testFactor(self):
        s = SupportVector([(1,1.0)])
        s.factor = 0.5
        self.assertEqual(str(s), "0.500000*SupportVector({1: 1.0})")
    
if __name__ == '__main__':
    unittest.main()
