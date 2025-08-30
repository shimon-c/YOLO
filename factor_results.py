import numpy as np
import pandas as pd
import argparse

class LinReg:
    def __init__(self, X,y):
        self.w = 0
        self.solve(X=X, y=y)

    def solve(self, X,y):
        XN = np.ones((X.shape[0],2))
        XN[:,0] = X
        XNN = np.linalg.inv(XN.T@XN)
        P_X = XNN@XN.T
        self.w = P_X @ y

    def predict(self,X):
        XN = np.ones((X.shape[0], 2))
        XN[:, 0] = X
        y = XN@self.w
        return y

def parse_args():
    ap = argparse.ArgumentParser("Factor_result")
    ap.add_argument("--csv_path", type=str,required=True, help="csv path and results")
    args = ap.parse_args()
    return args

def process(csv_path ,factor_score=85, max_limit_score=100):
    df = pd.read_csv(csv_path)
    col_names = df.columns.to_list()
    min_score = df[col_names[1]].min()
    max_score = df[col_names[1]].max()
    rng = max_score - min_score
    rng1 = max_score - factor_score
    factor = factor_score/min_score
    #df[col_names[1]] *= factor
    N = df.shape[0]
    X = df[col_names[1]].to_numpy()
    Y = X.copy()
    max_ids = X == max_score
    Y[max_ids] = max_limit_score
    min_ids = Y <= factor_score
    Y[min_ids] = factor_score
    lin_reg = LinReg(X=X, y=Y)
    for k in range(N):
        score = df.iloc[k,1]
        score = factor_score + max_score*(score-min_score) / rng
        #score = factor_score + (score - factor_score)/rng1
        if score>max_limit_score:
            score = max_limit_score
        print(f'{df.iloc[k,0]}\t {score}')
    ny = lin_reg.predict(X)
    print('\n--------------------\n')
    for k in range(N):
        name = df.iloc[k,0]
        print(f'{name}\t{ny[k]}')



if __name__ == "__main__":
    args = parse_args()
    process(csv_path=args.csv_path)