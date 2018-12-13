import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import itertools

def getDataSet(path):
    pd.set_option('display.max_colwidth', -1)

    dataset = pd.read_csv(path)
    display(dataset.head(n = 5))

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA'], axis = 1)
    return dataset, features, labels

def train_predict(learner, beta_value, X_train, y_train, X_test, y_test, dfResults):
    start = time()
    learner = learner.fit(X_train, y_train)
    end = time()

    train_time = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time

    pred_time = end - start

    f_train = fbeta_score(y_train, predictions_train, beta_value)

    f_test =  fbeta_score(y_test, predictions_test, beta_value)

    print("%s trained." % (learner.__class__.__name__))

    dfResults = dfResults.append({'learner': learner.__class__.__name__, 'train_time': train_time, 'pred_time': pred_time, 'f_test': f_test, 'f_train':f_train}, ignore_index=True)
    return learner, dfResults

def plotTimes(df, ax, time_field):
  for iLearner, row in df.iterrows():
    ax.bar(row['size_index'] + row['learner_index']*bar_width, row[time_field], width = bar_width, color = colors[row['learner']])
    ax.set_xticks([0.45, 1.45, 2.45])
    ax.set_xticklabels(["1%", "10%", "100%"])
    ax.set_xlabel("Training Set Size")
    ax.set_xlim((-0.1, 3.0))

def plotEval(df, ax, eval_field):
  learners = df['learner'].drop_duplicates().values.tolist()
  for learner in learners:
    dfLearner = df.loc[df['learner'] == learner].sort_values(by=['size_index'])
    size_index = dfLearner['size_index'].values.tolist()
    values = dfLearner[eval_field].values.tolist()

    ax.plot(size_index, values, color = colors[learner])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1%", "10%", "100%"])
    ax.set_xlabel("Training Set Size")

def tuneClassifier(clf, parameters, X_train, X_test, y_train, y_test):
  # TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
  from sklearn.metrics import make_scorer
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import ExtraTreesClassifier

  c, r = y_train.shape
  labels = y_train.values.reshape(c,)

  scorer = make_scorer(fbeta_score, beta=2)
  grid_obj = GridSearchCV(clf, param_grid=parameters,  scoring=scorer)
  grid_fit = grid_obj.fit(X_train, labels)
  best_clf = grid_fit.best_estimator_
  predictions = (clf.fit(X_train, labels)).predict(X_test)
  best_predictions = best_clf.predict(X_test)

  cnf_matrix = confusion_matrix(y_test, best_predictions)
  plot_confusion_matrix(cnf_matrix, classes=['Life not as risk', 'Life at risk'], normalize = True)
  print "Unoptimized model\n------"
  print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 2))
  print "\nOptimized Model\n------"
  print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 2))
  return best_clf




def scaleModel(clf, scaler, features, labels):
    transformer = MaxAbsScaler().fit(features)
    X = transformer.transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.20, random_state = 10)

    clf_Scaled = (clone(clf)).fit(X_train, y_train)
    predictions = clf_Scaled.predict(X_test)
    fb_score =  fbeta_score(y_test, predictions, beta_value)

    print "Scale results for {} with {}.".format(clf.__class__.__name__, scaler.__class__.__name__)
    print "X.shape {}.".format(X.shape)
    print "X_train.shape {}.".format(X_train.shape)
    print "X_test.shape {}.".format(X_test.shape)
    print "f-score {}.".format(fb_score)
    print "\n"
    return clf_Scaled, fb_score


from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

def scaleClassifier(clf, scalers, features, labels):
    scaledClassifier = (clone(clf))
    clf_f_score = 0
    for scaler in scalers:
        clfScaler, f_score = scaleModel(clf, scaler, features, labels)
        if(f_score > clf_f_score):
            scaledClassifier = clfScaler
            rf_f_score = f_score
    return scaledClassifier

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
    else:
        title = 'Confusion matrix, without normalization'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
