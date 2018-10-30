
import sys
sys.path.insert(0, '../src/')


import pandas as pd
import numpy as np
import time

from category_encoders import *
from preprocess import Preprocess


class PreprocessTest:
    def encodeFeatureTest(self, df):
        start_time = time.time()
        preprocess = Preprocess()
        preprocess.encodeFeatures(preprocess.likeliHood, df, ['AFEC_DPTO', 'AFEC_EDADR'], 'RIESGO_VIDA')
        print("--- %s seconds ---" % (time.time() - start_time))

    def testEncoders(self, df):
        start_time = time.time()
        y = df[['RIESGO_VIDA']]
        X = df.drop(['RIESGO_VIDA'], axis = 1)
        enc = TargetEncoder(cols=['AFEC_DPTO', 'AFEC_EDADR'], return_df = True).fit(X, y)
        numeric_dataset = enc.transform(X)
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    dataset = pd.read_csv("../datasets/dataset_clean.csv.gz", compression='gzip')
    dataset['RIESGO_VIDA'] = np.where(dataset['RIESGO_VIDA'] == 'si', 1,0)
    dataset = dataset.drop(['Unnamed: 0'], axis = 1)

    preprocessTest = PreprocessTest()
    preprocessTest.encodeFeatureTest(dataset)
    preprocessTest.testEncoders(dataset)
