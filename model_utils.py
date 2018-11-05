def suffle_split():
    # Import train_test_split


    # Split the 'features' and 'labels' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size = 0.20,
                                                        random_state = 10)

    # Show the results of the split
    print "features_final set has {} samples.".format(features.shape[0])
    print "Training set has {} samples.".format(X_train.shape[0])
    print "Testing set has {} samples.".format(X_test.shape[0])

    return X_train, X_test, y_train, y_test

def naive_predictor():
    '''
    TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data
    encoded to numerical values done in the data preprocessing step.
    FP = income.count() - TP # Specific to the naive case

    TN = 0 # No predicted negatives in the naive case
    FN = 0 # No predicted negatives in the naive case
    '''

    tp = float(np.sum(labels['RIESGO_VIDA']))
    fp = float(labels['RIESGO_VIDA'].count() - tp)
    tn = 0
    fn = 0

    # TODO: Calculate accuracy, precision and recall
    accuracy = (tp + tn)/labels['RIESGO_VIDA'].count()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(accuracy)
    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    # HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    beta = 2
    fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    # Print the results
    return accuracy, fscore

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time

    # TODO: Calculate the training time
    results['train_time'] = end - start

    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start

    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] =  accuracy_score(y_test, predictions_test)

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    b=0.5
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, b)

    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] =  fbeta_score(y_test, predictions_test, b)

    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # Return the results
    return results

  def eval_learners(clf_A, clf_B, clf_C, X_train, y_train, X_test, y_test):

      samples_100 = len(y_train)
      samples_10 = int(samples_100*0.1)
      samples_1 = int(samples_10*0.1)
      # Collect results on the learners
      results = {}
      for clf in [clf_A, clf_B, clf_C]:
          clf_name = clf.__class__.__name__
          results[clf_name] = {}
          for i, samples in enumerate([samples_1, samples_10, samples_100]):
              results[clf_name][i] = \
              train_predict(clf, samples, X_train, y_train, X_test, y_test)

        # Run metrics visualization for the three supervised learning models chosen
        vs.evaluate(results, accuracy, fscore)
