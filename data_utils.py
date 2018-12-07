import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

def getFeaturesDistribution(features):
    dfColumns = pd.DataFrame(columns=['Feature','Distinct Values'])
    for colName in features.columns.values:
        dfColumns = dfColumns.append({'Feature': colName, 'Distinct Values': features[colName].unique().size}, ignore_index=True)
    return dfColumns.infer_objects()

def showFeaturesDistribution(dfColumns):
    dfFeatures = dfColumns.copy(deep=True)
    dfFeatures.set_index("Feature",drop=True,inplace=True)
    dfFeatures.plot( kind='bar', figsize = (15,5))

def getSample(mX_train, mY_train, category, sample_size):
    images = []
    total = 0
    idx = 0
    while total < sample_size:
        if(mY_train[idx][1] == float(category)):
            images.append(mX_train[idx])
            total += 1
        idx += 1
    return images

def scale_number(x, xMin, xMax, a, b):
    return ((x - xMin) / (xMax - xMin))*(b - a) + a

def plotFeatures(images, columns):
    plt.figure(figsize=(5,5))
    average = []
    for i, image in enumerate(images):
        plt.subplot(5, 4, (i+1))
        #plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)
        average.append(np.average(image))

    print("Average gray scale: %f"%(scale_number(np.average(average), 0, 1, 0, 255)))
