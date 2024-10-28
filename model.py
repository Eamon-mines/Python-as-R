import numpy as np
import scipy.stats
import pandas as pd
"""
THE PURPOSE OF THIS CLASS IS TO BE GIVEN SOME MATRIX AND PRODUCE AN LM MODEL.
IT WILL HOLD THE COEFFICIENTS, RESIDUALS, ERRORS, AND OTHER INFO FROM THE R LM FUNCTION.
"""
class lm_model:
    def __init__(self):
        self.betaHat: np.array
        self.xArray: np.array
        self.yArray: np.array
        self.residuals: np.array
        self.std_err: np.array
        self.tvalue: np.array
        self.probGreaterTvalue: np.array
        self.intercept = 0
        self.degreeFree = 0
        self.residError = 0
        self.rank = 0
        self.rsquared = 0
        self.adjustedrsquared = 0
        self.FStat = 0
        self.FProb = 0

    def set_from_list(self, y: list , x: list , intercept=0):
        ''' y = the list or array of y values that we are predicting\n
            x = the list or array of x values that we are predicting\n
            intercept = 0 DEFAULT. When 0, you want a predicted value for the intercept.\n
            A value of 1, means no intercept prediction.\n\n
            
            Example Usage:\n
            x = [ [1, 2, 3],    # the one in the beginning, is for the intercept
                  [1, 4, 5],
                  [1, 5, 6]]
                  
            y = [[1],
                 [2],
                 [3]]
                 
            model = mod.make(y, x)'''

        if intercept not in [0, 1]:
            raise ValueError("Intercept value needs to be 0 or 1")
        
        if intercept == 0:
            for i in range(len(y)):
                x[i] = [1] + x[i]


        y = np.array(y)

        x = np.array(x)


        if x.shape[0] != y.shape[0]:   # if wrong number of (x,y) pairs, kill prog
            raise TypeError(f"X and Y dont have the same number of rows")
        
        self.xArray = x
        self.yArray = y

    
    def assign_variables(self):
        self.std_error()
        self.t_value()
        self.degrees_of_Freedom()
        self.residual_standard_error()
        self.calc_rsquared()
        self.calc_adjustRSquared()
        self.calc_FStat()
        self.calc_probs()

    def calculate_betas(self):
        xTranspose = self.xArray.transpose()
        xTx = np.matmul(xTranspose, self.xArray)
        xTy = np.matmul(xTranspose, self.yArray)
        self.betaHat = np.matmul(np.linalg.inv(xTx), xTy)
    
    def calculate_residuals(self):
        self.residuals = np.subtract(self.yArray, np.matmul(self.xArray, self.betaHat))

    def set_from_pandas(self, y: pd.DataFrame, x: pd.DataFrame, intercept=0):
        if intercept not in [0, 1]:
            raise ValueError("Intercept value needs to be 0 or 1")
        
        if intercept == 0:
            ones = []
            for i in range(len(y)):
                ones.append(1)
            x.insert(0, "Intercept", ones)


        y = np.array(y.to_numpy())

        x = np.array(x.to_numpy())


        if x.shape[0] != y.shape[0]:   # if wrong number of (x,y) pairs, kill prog
            raise TypeError(f"X and Y dont have the same number of rows")
        
        self.xArray = x
        self.yArray = y


    def std_error(self):
        n= self.xArray.shape[0]
        p = self.xArray.shape[1]

        sigmahat = (np.sum(self.residuals**2)/(n-p))**0.5
        


        covBeta = sigmahat**2 * np.linalg.inv(np.matmul(np.transpose(self.xArray), self.xArray))


        self.std_err = np.sqrt(np.diag(covBeta))

    def t_value(self):
        t_vals = []
        for i in range(len(self.betaHat)):
            t_vals.append([self.betaHat[i][0]/self.std_err[i]])
        self.tvalue = np.array(t_vals)
        

    def degrees_of_Freedom(self):
        n= self.xArray.shape[0]
        p = self.xArray.shape[1]
        self.degreeFree = n - p
        self.rank = p

    def residual_standard_error(self):
        k = self.betaHat.shape[0] - 1
        sse = np.sum(self.residuals**2)
        n = self.residuals.shape[0]
        self.residError = (sse / (n - (1 + k)))**.5

    def calc_rsquared(self):
        sumSquaredy = np.sum((np.subtract(self.yArray, np.mean(self.yArray)))**2)
        sse = np.sum(self.residuals**2)
        self.rsquared = (sumSquaredy - sse)/sumSquaredy

    def calc_adjustRSquared(self):
        n = self.yArray.shape[0]
        k = self.betaHat.shape[0]
        sse = np.sum(self.residuals**2)
        SumSquaredy = np.sum((np.subtract(self.yArray, np.mean(self.yArray)))**2)
        self.adjustedrsquared = 1 - (sse / SumSquaredy) * (n - 1) / ( n - (k + 1))

    def calc_FStat(self):
        n = self.yArray.shape[0]
        sse = np.sum(self.residuals**2)
        sumSquarey = np.sum((np.subtract(self.yArray, np.mean(self.yArray)))**2)
        k = self.betaHat.shape[0] - 1
        self.FStat = ((sumSquarey - sse)/k) / (sse/(n - (k + 1))) 
    
    def calc_probs(self):
        d = self.yArray.shape[0] - 1
        n = self.betaHat.shape[0] - 1
        # calc t prob
        tValProbs = []
        for value in self.tvalue:
            tValProbs.append(scipy.stats.t.sf(value[0], d))

        self.probGreaterTvalue = np.array(tValProbs)

        # calc f prob
        p_val = scipy.stats.f.sf(self.FStat, n, d)
        self.FProb = p_val

        


    def print_betaHat(self):
        print("Estimate:\n")
        if self.intercept == 0:   # if intercept exists
            print("Intercept: ", end="")
            for i in range(self.betaHat.shape[0]):
                if i == 0:
                    print(format(self.betaHat[i][0], "e"))
                else:
                    print(f"       X{i}: {self.betaHat[i][0]:e}")
        else:    # if no intercept wanted
            for i in range(self.betaHat.shape[0]):
                print(f"X{i}: {self.betaHat[i][0]:e}")
        print("\n")
    
    def print_stdError(self):
        print("Std. Error:\n")
        if self.intercept == 0:   # if intercept exists
            print("Intercept: ", end="")
            for i in range(self.std_err.shape[0]):
                if i == 0:
                    print(format(self.std_err[i], "e"))
                else:
                    print(f"       X{i}: {self.std_err[i]:e}")
        else:    # if no intercept wanted
            for i in range(self.std_err.shape[0]):
                print(f"X{i}: {self.std_err[i]:e}")
        print()