import pandas as pd
import numpy as np

class Preprocess:
    def likeliHood(self, positives, negatives):
        if((positives + negatives) == 0):
            return 0
        return positives / (positives + negatives)

    def weightOfEvidence(self, positives, negatives):
        math.log(positives/negatives)*100

    def encodeFeature(self, f, df, featureColumn, targetColumn):
        values = df[[featureColumn]].drop_duplicates().astype('str').values.flatten()
        for value in values:
            totalPositive = float(len(df[(df[featureColumn].astype('str') == value) & (df[targetColumn] == 1)].index))
            totalNegative = float(len(df[(df[featureColumn].astype('str') == value) & (df[targetColumn] == 0)].index))
            df.loc[df[featureColumn] == value, featureColumn] = f(totalPositive, totalNegative)
        return df

    def encodeFeatures(self, f, df, features, targetColumn):
        for feature in features:
            print("Encoding: %s" % feature)
            df = self.encodeFeature(f, df, feature, targetColumn)
        return df
