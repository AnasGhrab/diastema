import unittest
from diastema import intervalles as di
from numpy import log2,log10

class Test_intervalles(unittest.TestCase):
    
    def test_cent_converter(self):
        inter = 4/3.
        inter_sav = log10(inter)*1000
        self.assertEqual(inter_sav, di.savart(inter))

    def test_set_unit_arguments(self):
        unit = "coucou"
        self.assertRaises(ValueError, di.set_unit, unit )

if __name__ == '__main__':
    unittest.main()