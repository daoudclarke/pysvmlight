# Bismillahi-r-Rahmani-r-Rahim

import unittest

from svmlight import SupportVector, Document

class DocumentTestCase(unittest.TestCase):
    def testConstruction(self):
        self.assertEqual(str(Document(1,SupportVector([1,2,3]))),
                         "Document(1, SupportVector([1, 2, 3]))")

    def testConstructionNeedsSupportVector(self):
        self.assertRaises(TypeError, Document, 2, [1,2,3])
    
if __name__ == '__main__':
    unittest.main()
