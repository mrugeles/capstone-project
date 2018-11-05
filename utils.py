from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from category_encoders import *
import time

def getFeaturesDistribution(features):
    dfColumns = pd.DataFrame(columns=['Feature','Distinct Values'])
    for colName in features.columns.values:
        dfColumns = dfColumns.append({'Feature': colName, 'Distinct Values': features[colName].unique().size}, ignore_index=True)
    return dfColumns.infer_objects()

def showFeaturesDistribution(dfColumns):
    dfFeatures = dfColumns.copy(deep=True)
    dfFeatures.set_index("Feature",drop=True,inplace=True)
    dfFeatures.plot( kind='bar', figsize = (15,5))

def encode(features, labels):
    start_time = time.time()
    enc = TargetEncoder(cols=features.columns.values.tolist(), return_df = True).fit(features, labels)
    dataset_encoded = enc.transform(features)
    print("--- %s seconds ---" % (time.time() - start_time))
