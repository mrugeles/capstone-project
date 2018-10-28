
import sys
sys.path.insert(0, '../src/')

from preprocess import Preprocess
import pandas as pd

class PreprocessTest:
    def encodeFeatureTest(self):
        data = [
                ['Up',1],
                ['Up',1],
                ['Down',0],
                ['Flat',0],
                ['Down',1],
                ['Up',0],
                ['Down',0],
                ['Flat',0],
                ['Flat',1],
                ['Flat',1]
                ]
        df = pd.DataFrame(data,columns=['Trend','Target'])
        print(len(df[(df['Trend'] == 'Up') & (df['Target'] == 1)]))
        preprocess = Preprocess()
        print(preprocess.encodeFeatures(preprocess.likeliHood, df, ['Trend'], 'Target'))

if __name__ == '__main__':
    preprocessTest = PreprocessTest()
    preprocessTest.encodeFeatureTest()
