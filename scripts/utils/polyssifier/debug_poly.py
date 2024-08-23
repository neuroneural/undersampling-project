def poly(data, label, n_folds=10, scale=True, exclude=[],
         feature_selection=False, save=False, scoring='auc',
         project_name='', concurrency=1, verbose=True, random_state=1988):
    '''
    Input
    data         = numpy matrix with as many rows as samples
    label        = numpy vector that labels each data row
    n_folds      = number of folds to run
    scale        = whether to scale data or not
    exclude      = list of classifiers to exclude from the analysis
    feature_selection = whether to use feature selection or not (anova)
    save         = whether to save intermediate steps or not
    scoring      = Type of score to use ['auc', 'f1']
    project_name = prefix used to save the intermediate steps
    concurrency  = number of parallel jobs to run
    verbose      = whether to print or not results
    Ouput
    scores       = matrix with scores for each fold and classifier
    confusions   = confussion matrix for each classifier
    predictions  = Cross validated predicitons for each classifier
    '''
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

    assert label.shape[0] == data.shape[0],\
        "Label dimesions do not match data number of rows"
    
    _le = LabelEncoder()
    _le.fit(label)
    label = _le.transform(label)
    n_class = len(np.unique(label))
    logger.info('Detected ' + str(n_class) + ' classes in label')

    if save and not os.path.exists('poly_{}/models'.format(project_name)):
        os.makedirs('poly_{}/models'.format(project_name))

    logger.info('Building classifiers ...')
    classifiers = build_classifiers(exclude, scale,
                                    feature_selection,
                                    data.shape[1])

    scores = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [classifiers.keys(), ['train', 'test']]),
        index=range(n_folds))
    predictions = pd.DataFrame(columns=classifiers.keys(),
                               index=range(data.shape[0]))
    test_prob = pd.DataFrame(columns=classifiers.keys(),
                             index=range(data.shape[0]))
    confusions = {}
    coefficients = {}
    # !fitted_clfs =
    # pd.DataFrame(columns=classifiers.keys(), index = range(n_folds))

    logger.info('Initialization, done.')

    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    skf.get_n_splits(np.zeros(data.shape[0]), label)
    kf = list(skf.split(np.zeros(data.shape[0]), label))

    # Parallel processing of tasks
    
    manager = Manager()
    args = manager.list()
    args.append({})  # Store inputs
    shared = args[0]
    shared['kf'] = kf
    shared['X'] = data
    shared['y'] = label
    args[0] = shared

    args2 = []
    for clf_name, val in classifiers.items():
        for n_fold in range(n_folds):
            args2.append((args, clf_name, val, n_fold, project_name,
                          save, scoring))

    if concurrency == 1:
        result = list(starmap(fit_clf, args2))
    else:
        pool = Pool(processes=concurrency)
        result = pool.starmap(fit_clf, args2)
        pool.close()

    fitted_clfs = {key: [] for key in classifiers}

    # Gather results
    for clf_name in classifiers:
        coefficients[clf_name] = []
        temp = np.zeros((n_class, n_class))
        temp_pred = np.zeros((data.shape[0], ))
        temp_prob = np.zeros((data.shape[0], ))
        clfs = fitted_clfs[clf_name]
        for n in range(n_folds):
            train_score, test_score, prediction, prob, confusion,\
                coefs, fitted_clf = result.pop(0)
            clfs.append(fitted_clf)
            scores.loc[n, (clf_name, 'train')] = train_score
            scores.loc[n, (clf_name, 'test')] = test_score
            temp += confusion
            temp_prob[kf[n][1]] = prob
            temp_pred[kf[n][1]] = _le.inverse_transform(prediction)
            coefficients[clf_name].append(coefs)

        confusions[clf_name] = temp
        predictions[clf_name] = temp_pred
        test_prob[clf_name] = temp_prob

    # Voting
    fitted_clfs = pd.DataFrame(fitted_clfs)
    scores['Voting', 'train'] = np.zeros((n_folds, ))
    scores['Voting', 'test'] = np.zeros((n_folds, ))
    temp = np.zeros((n_class, n_class))
    temp_pred = np.zeros((data.shape[0], ))
    for n, (train, test) in enumerate(kf):
        clf = MyVoter(fitted_clfs.loc[n].values)
        X, y = data[train, :], label[train]
        scores.loc[n, ('Voting', 'train')] = _scorer(clf, X, y)
        X, y = data[test, :], label[test]
        scores.loc[n, ('Voting', 'test')] = _scorer(clf, X, y)
        temp_pred[test] = clf.predict(X)
        temp += confusion_matrix(y, temp_pred[test])

    confusions['Voting'] = temp
    predictions['Voting'] = temp_pred
    test_prob['Voting'] = temp_pred
    ######

    # saving confusion matrices
    if save:
        with open('poly_' + project_name + '/confusions.pkl', 'wb') as f:
            p.dump(confusions, f, protocol=2)

    if verbose:
        print(scores.astype('float').describe().transpose()
              [['mean', 'std', 'min', 'max']])
    
    return Report(scores=scores, confusions=confusions,
                predictions=predictions, test_prob=test_prob,
                coefficients=coefficients,
                feature_selection=feature_selection, target=label)
    
