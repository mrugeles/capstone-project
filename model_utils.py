import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import fbeta_score

def getDataSet(path):
    pd.set_option('display.max_colwidth', -1)

    dataset = pd.read_csv(path, compression='gzip')
    dataset = dataset.drop(['Unnamed: 0'], axis = 1)
    display(dataset.head(n = 5))

    labels = dataset[['RIESGO_VIDA']]
    features = dataset.drop(['RIESGO_VIDA', 'PQR_ESTADO'], axis = 1)
    return dataset, features, labels

def train_predict(learner, learner_index, size_index, sample_size, X_train, y_train, X_test, y_test, dfResults):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time

    # TODO: Calculate the training time
    train_time = end - start

    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # TODO: Calculate the total prediction time
    pred_time = end - start

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    b=2
    f_train = fbeta_score(y_train[:300], predictions_train, b)

    # TODO: Compute F-score on the test set which is y_test
    f_test =  fbeta_score(y_test, predictions_test, b)

    # Success
    print("%s trained on %d samples." % (learner.__class__.__name__, sample_size))

    dfResults = dfResults.append({'learner': learner.__class__.__name__, 'learner_index': learner_index, 'size_index': size_index, 'train_time': train_time, 'pred_time': pred_time, 'f_test': f_test, 'f_train':f_train}, ignore_index=True)
    # Return the results
    return dfResults

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

def plotResults(dfResults):
    plt.figure(figsize = (15,10))

    axTraining = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    axFscoreTraining = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    axPrediction = plt.subplot2grid((2, 3), (1, 0), colspan=1)
    axFscoreTest = plt.subplot2grid((2, 3), (1, 1), colspan=2)

    bar_width = 0.3
    colors = {
          'SGDClassifier': '#A00000',
          'AdaBoostClassifier': '#00A0A0',
          'RandomForestClassifier': '#00A000'
    }

    plotTimes(dfResults, axTraining, 'train_time')
    plotTimes(dfResults, axPrediction, 'pred_time')

    plotEval(dfResults, axFscoreTraining, 'f_train')
    plotEval(dfResults, axFscoreTest, 'f_test')

    # Add unique y-labels
    axTraining.set_ylabel("Time (in seconds)")
    axFscoreTraining.set_ylabel("F-score")
    axPrediction.set_ylabel("Time (in seconds)")
    axFscoreTest.set_ylabel("F-score")

    axTraining.set_title("Model Training")
    axFscoreTraining.set_title("F-score on Training Subset")
    axPrediction.set_title("Model Predicting")
    axFscoreTest.set_title("F-score on Testing Set")

    axFscoreTraining.set_ylim((0, 1))
    axFscoreTest.set_ylim((0, 1))

    # Create patches for the legend
    learners = dfResults['learner'].drop_duplicates().values.tolist()
    patches = []
    for i, learner in enumerate(learners):
        patches.append(mpatches.Patch(color = colors[learner], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (0.5, 2.43), loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')

    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.tight_layout()
    plt.show()

def tuneClassifier(clf, parameters, X_train, X_test, y_train, y_test):

  # TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
  from sklearn.metrics import make_scorer
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import ExtraTreesClassifier

  c, r = y_train.shape
  labels = y_train.values.reshape(c,)

  # TODO: Make an fbeta_score scoring object using make_scorer()
  scorer = make_scorer(fbeta_score, beta=2)
  # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
  grid_obj = GridSearchCV(clf, param_grid=parameters,  scoring=scorer)
  # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
  grid_fit = grid_obj.fit(X_train, labels)
  # Get the estimator
  best_clf = grid_fit.best_estimator_
  # Make predictions using the unoptimized and model
  predictions = (clf.fit(X_train, labels)).predict(X_test)
  best_predictions = best_clf.predict(X_test)
  # Report the before-and-afterscores
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
    fb_score =  fbeta_score(y_test, predictions, 2)

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