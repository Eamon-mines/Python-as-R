import sys, os
import csv
import numpy as np
import pandas as pd
import unittest
def assign_model():
        df = pd.read_csv("./tests/data.csv")
        ys = df.pMean
        xs = df[["elevation", "lon", "lat"]]
        mod = m.lm_model()
        mod.set_from_pandas(ys, xs)
        mod.calculate_betas()
        mod.calculate_residuals()
        mod.assign_variables()
        return (ys, xs, mod)

class TestLMModel(unittest.TestCase):
    def test_assign_x_y(self):
        '''
        If this test is wrong, need to go and edit pandas to np matrix. If good, then lets go.
        '''
        x = []
        y = []
        with open("./tests/data.csv", "r") as file:
            read = csv.reader(file)
            for line in read:
                if line[0] == 'pMean':
                    continue
                xrow = []
                y.append(float(line[0]))
                xrow.append(float(line[3]))
                xrow.append(float(line[1]))
                xrow.append(float(line[2]))
                x.append(xrow)
        mod = m.lm_model()
        mod.set_from_list(y, x)

        _, _, mod2 = assign_model()
        
        for i in range(len(x)):
            np.testing.assert_array_equal(mod.xArray[i],mod2.xArray[i], "Xs are same for inputs")
        np.testing.assert_array_equal(mod.yArray,mod2.yArray, "Ys are same for inputs")
    
    def test_betas_against_true(self):
        _, _, mod = assign_model()


        # NOTE: all numbers are pulled directly from R for the data.
        np.testing.assert_almost_equal(85.42509, mod.betaHat[0], 5, "Test intercept from known data")  # test that intercept is same
        np.testing.assert_almost_equal(0.0003954287, mod.betaHat[1], 10, "Test for elevation")
        np.testing.assert_almost_equal(0.8725536, mod.betaHat[2], 7, "Test for longitude")
        np.testing.assert_almost_equal(0.2487488, mod.betaHat[3], 7, "Test for latitude")

    def test_residuals(self):
        _, _, mod = assign_model()

        resids = []
        with open("./tests/residuals.csv", "r") as file:
            read = csv.reader(file)
            for line in read:
                if "x" in line:
                    continue
                resids.append(float(line[1]))
        
        self.maxDiff = None
        np.testing.assert_almost_equal(resids, mod.residuals, 7, "Residuals dont match.")
    
    def test_rsquared_and_adjust(self):
        _, _, mod = assign_model()

        np.testing.assert_almost_equal(.5481, mod.rsquared, 4)
        np.testing.assert_almost_equal(.543, mod.adjustedrsquared, 3)

    def test_rank(self):
        _, _, mod = assign_model()
        np.testing.assert_equal(4, mod.rank)
    
    def test_resid_std_err(self):
        _, _, mod = assign_model()
        np.testing.assert_almost_equal(1.232, mod.residError, 3)



if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import model as m
    unittest.main()