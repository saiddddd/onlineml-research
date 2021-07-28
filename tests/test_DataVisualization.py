from tools.DataVisualization import CalibrationPlot

def test_add_calibration_curve():

    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import calibration_curve

    cplot = CalibrationPlot()

    X, y = datasets.make_classification(n_samples=100000, n_features=20, n_informative=2, n_redundant=2)

    train_samples = 100

    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = y[:train_samples]
    y_test = y[train_samples:]

    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC()
    rfc = RandomForestClassifier()

    for clf, name in [(lr, 'Logistic'),
                      (gnb, 'Naive Bayes'),
                      (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1] # only take the positive prob
        else: # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        cplot.add_calibration_curve(mean_predicted_value, fraction_of_positives, label="%s" % (name, ))
        cplot.add_histogram(prob_pos, name)

    cplot.save_fig()

    assert True
