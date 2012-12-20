# Bismillahi-r-Rahmani-r-Rahim

import unittest

from svmlight import SupportVector, Document, DocumentFactory

class DocumentFactoryTestCase(unittest.TestCase):
    def testDocumentNums(self):
        factory = DocumentFactory()
        d1 = factory.new([1,2,3])
        d2 = factory.new([4,5,6])
        self.assertEqual(1, d1.docnum)
        self.assertEqual(2, d2.docnum)

    def testDocumentVector(self):
        factory = DocumentFactory()
        d1 = factory.new(['a','b','c'])
        d2 = factory.new(['b','d','e'])
        print factory.nums
        self.assertEqual('Document(1, SupportVector({1: 1.0, 2: 1.0, 3: 1.0}))', str(d1))
        self.assertEqual('Document(2, SupportVector({2: 1.0, 4: 1.0, 5: 1.0}))', str(d2))
    
if __name__ == '__main__':
    unittest.main()
