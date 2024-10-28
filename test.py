import model as m
import csv
import pandas as pd
import unittest


class TestLMModel(unittest.TestCase):
    def test_assign_x_y(self):
        '''
        If this test is wrong, need to go and edit pandas to np matrix. If good, then lets go.
        '''
        x = []
        y = []
        with open("test.csv", "r") as file:
            read = csv.reader(file)
            for line in read:
                if line[0] == 'pMean':
                    continue
                xrow = []
                y.append([float(line[0])])
                #xrow.append(1)
                xrow.append(float(line[3]))
                xrow.append(float(line[1]))
                xrow.append(float(line[2]))
                x.append(xrow)
        mod = m.lm_model()
        mod.set_from_list(y, x)

        mod2 = m.lm_model()
        df = pd.read_csv("test.csv")
        ys = df.pMean
        xs = df[["elevation", "lon", "lat"]]
        mod2.set_from_pandas(ys, xs)
        for i in range(len(x)):
            self.assertCountEqual(mod.xArray[i],mod2.xArray[i], "Xs are same for inputs")
        self.assertCountEqual(mod.yArray,mod2.yArray, "Ys are same for inputs")
    
    def test_betas_against_true(self):



if __name__ == '__main__':
    unittest.main()