import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from handy import df_preprossecing, getFeatureList, getSensorList, chooseFeatures, combineSensors
from scipy.stats.mstats import kruskalwallis
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, RFECV, f_classif, \
    mutual_info_classif, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier
from boruta import BorutaPy

# All Feature Selection codes used. All modes takes in a dataframe where 'Damage' is the class label,
# and will return a new dataframe where features are eliminated. Additionally, all eliminated features
# can be printed. Some functions can also return the score of features and eliminated features.

def thresholdEliminate(rankings, threshold, greaterThan=True):
    """Eliminates features outside threshold. Implemented in several functions.
        input:
            rankings - series of features and their ranks
            threshold - Set threshold
            greaterThan - Bool, if the requirement is that they are greater than the threshold"""
    if greaterThan == False:
        eliminate = rankings[rankings >= threshold]
        keep = rankings[rankings < threshold]
    else:
        eliminate = rankings[rankings <= threshold]
        keep = rankings[rankings > threshold]
    return keep, eliminate

def ML_method(method):  # LogReg, DecisionTree, RandomForest, ExtremeTree, XGB
    """Several defined machine learning methods"""
    if method == 'LogReg':
        clf = LogisticRegression(max_iter=150, tol=1e-4, C=5, verbose=0)
    elif method == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif method == 'RandomForest':
        clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=4, min_samples_leaf=2,
                                     min_samples_split=5,  n_estimators=100, n_jobs=1)  # , max_features=1
    elif method == 'ExtremeTree':
        clf = ExtraTreesClassifier()
    elif method == 'XGB':
        clf = XGBClassifier(min_child_weight=1, subsample=0.65, max_depth=5, learning_rate=0.4, gamma=0.4,
                            reg_lambda=3, colsample_bytree=0.3, n_estimators=50,  use_label_encoder=False,
                            n_jobs=1, cv=5, verbosity=0)
    return clf

# ----------------Filter Methods-----------------

def check_variance_features(df, threshold=0.0, showElim=False):
    '''Remove constant and semi-constant features.
    Input:
        df: Dataframe with features and damage classes
        threshold: Variance must be higher than'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    var_tresh = VarianceThreshold(threshold=threshold)
    var_tresh.fit(X)
    eliminate = X.columns[var_tresh.get_support()==False]
    keep = X.columns[var_tresh.get_support()]
    if showElim == True:
        print('Eliminated by Constant: %i'%len(eliminate))
        print(eliminate)
    X = X[keep]
    X['Damage'] = y
    return X

def anova_FS(df, p_value=0.05, showElim=False):
    '''Anova filter method with p-value as threshold.
    Input:
        df: Dataframe with features and damage classes
        p-value: features with p-value higher than threshold eliminated.'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    sel = f_classif(X, y)  # Returns the f-score and the p-value
    p_values = pd.Series(sel[1], index=X.columns)
    p_values.sort_values(ascending=True, inplace=True)
    keep, eliminate = thresholdEliminate(p_values, p_value, greaterThan=False)
    if showElim == 1:
        print('Eliminated by ANOVA: %i'%len(eliminate.index))
        print(eliminate.index)
    X = X[keep.index]
    X['Damage'] = y
    return X

def kruskal_wallis(df, p_value=0.05, showElim=False, segments = 50):
    '''Kruskal Wallis filter method with p-value as threshold.
    Input:
        df: Dataframe with features and damage classes
        p-value: features with p-value higher than threshold eliminated.
        segments: number of splits in each dataseries'''
    def f(feat,n, segments):
        return np.array(feat[n*segments:(n+1)*segments])
    X = df.drop(['Damage'], 1)
    y = df.Damage
    p_values = np.zeros(X.shape[1])
    # Get the p-values using Kruskal Wallis
    for i in range(X.shape[1]):
        temp = X[X.columns[i]]
        p_values[i]=kruskalwallis(f(temp,0,segments),f(temp,1,segments),f(temp,2,segments),
        f(temp,3,segments),f(temp,4,segments),f(temp,5,segments),f(temp,6,segments)).pvalue
    p_values = pd.Series(p_values,index=X.columns)
    p_values.sort_values(ascending=True, inplace=True)
    keep, eliminate = thresholdEliminate(p_values, p_value, greaterThan=False)
    if showElim == 1:
        print('Eliminated by Kruskal Wallis: %i'%len(eliminate.index))
        print(eliminate.index)
    X = X[keep.index]
    X['Damage'] = y
    return X

def uni_correlation(df, threshold=0.1, method='pearson', absVal=True, showElim=False):
    '''Unicorrelation filter method with p-value as threshold.
    Input:
        df: Dataframe with features and damage classes
        threshold: Correlation coefficient must be higher than
        method: Correlation method. Choose between 'pearson', 'spearman'
        absVal: Absolute value of the correlation coefficient'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    uni_corr = X.corrwith(y,method=method)
    if absVal == True:
        uni_corr = abs(uni_corr)
    uni_corr = uni_corr.sort_values(ascending=False)
    keep, eliminate = thresholdEliminate(uni_corr, threshold, greaterThan=True)
    if showElim == 1:
        print('Eliminated by univariate correlation: %i'%len(eliminate.index))
        print(eliminate.index)
    X = X[keep.index]
    X['Damage'] = y
    return X, uni_corr

def mutual_information_gain_FS(df, threshold=0.1, showElim=False):
    '''Mutual information filter method
    Input:
        df: Dataframe with features and damage classes
        threshold: mutual information must be higher than'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    mig = mutual_info_classif(X, y)
    mig = pd.Series(mig, index=X.columns)
    mig.sort_values(ascending=False, inplace=True)
    keep, eliminate = thresholdEliminate(mig, threshold, greaterThan=True)
    if showElim == 1:
        print('Eliminated by Mutual Information Gain: %i'%len(eliminate.index))
        print(eliminate.index)
    X = X[keep.index]
    X['Damage'] = y
    return X, mig

def ROC_AUC(df, threshold=0.1, ML='RandomForest', testSize=0.5, avgMethod='macro', multiClass='ovo', auc_mod=True,
            showElim=False):
    '''AUC filter method
    Input:
        df: Dataframe with features and damage classes
        threshold: AUC must be higher than
        ML: Machine Learning method to use. Select between 'LogReg', 'DecisionTree',
                                'RandomForest', 'ExtremeTree', 'XGB'
        testSize: Fraction of training data
        avgMethod: AUC averaging method - Select between 'macro', 'weighted'
        multiClass: see roc_auc_score documentation
        auc_mod: score becomes abs(0.5 - auc)'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, stratify=y)  # , random_state=10
    auc = []
    for feature in X_train.columns:
        clf = ML_method(ML)
        clf.fit(X_train[feature].to_frame(), y_train)
        y_pred_proba = clf.predict_proba(X_test[feature].to_frame())
        auc.append(roc_auc_score(y_test, y_pred_proba, multi_class=multiClass, average=avgMethod))
    auc_values = pd.Series(auc, index=X.columns)
    if auc_mod == True:
        auc_values = abs(0.5 - auc_values)
    auc_values.sort_values(ascending=False, inplace=True)
    keep, eliminate = thresholdEliminate(auc_values, threshold, greaterThan=True)
    if showElim == True:
        print('Eliminated by AUC-ROC: %i'%len(eliminate.index))
        print(eliminate.index)
    X = X[keep.index]
    X['Damage'] = y
    return X

def model_based_ranking(df, threshold=0.3, ML='RandomForest', testSize=0.5, showElim=False):
    '''Model based ranking filter method
    Input:
        df: Dataframe with features and damage classes
        threshold: predictive accuracy must be higher than
        ML: Machine Learning method to use. Select between 'LogReg', 'DecisionTree',
                                'RandomForest', 'ExtremeTree', 'XGB'
        testSize: Fraction of training data'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, stratify=y)  # , random_state=10
    ranking = []
    for feature in X_train.columns:
        clf = ML_method(ML)
        clf.fit(X_train[feature].to_frame(), y_train)
        y_pred = clf.predict(X_test[feature].to_frame())
        ranking.append(accuracy_score(y_test, y_pred))
    ranking = pd.Series(ranking, index=X.columns)
    ranking.sort_values(ascending=False, inplace=True)
    keep, eliminate = thresholdEliminate(ranking, threshold, greaterThan=True)
    if showElim == True:
        print('Eliminated by Model Based Ranking: %i'%len(eliminate.index))
        print(eliminate.index)
    X = X[keep.index]
    X['Damage'] = y
    return X, ranking

def find_correlated_features(corrmat, threshold):
    '''Create list with features that have correlation higher than threshold with at least one other feature'''
    corrmat=abs(corrmat)
    col_corr = set()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if (corrmat.iloc[i, j]) > threshold and i!=j:
                colname1 = corrmat.columns[i]
                colname2 = corrmat.columns[j]
                col_corr.add(colname1)
                col_corr.add(colname2)
    return col_corr
def corr_grouping(corrmat, threshold, showGroups=False):
    '''Create groups of features which correlate'''
    corrdata = corrmat.abs().stack()
    corrdata = corrdata.sort_values(ascending=False)
    corrdata = corrdata[corrdata>threshold]
    corrdata = corrdata[corrdata<1]
    corrdata = pd.DataFrame(corrdata).reset_index()
    corrdata.columns = ['feature1','feature2','corr_value']
    grouped_feature_list = []
    correlated_groups_list = []
    for feature in corrdata.feature1.unique():
        if feature not in grouped_feature_list:
            correlated_block = corrdata[corrdata.feature1 == feature]
            grouped_feature_list = grouped_feature_list + list(correlated_block.feature2.unique())
            correlated_groups_list.append(correlated_block)
    if showGroups==True:
        for group in correlated_groups_list:
            print('Groups:')
            print(group)
        print('Total Groups: %i'%(len(correlated_groups_list)))
    return correlated_groups_list
def corr_group_feature_importance(correlated_groups_list, corr_features, X, y, showSelection=False):
    '''Keep the best feature in each group.
    Can select method to rank features inside the function.
    showSelection: shows the the discarded and selected feature for each group'''
    important_features = []
    for group in correlated_groups_list:
        features = list(group.feature1.unique()) + list(group.feature2.unique())
        df_group = X[features]
        df_group['Damage'] = y
            # method1 = 'mig' , 'auc' , 'modelBased' , 'treeBased'
        df_group = getBestFeature(df_group,method1='mig',method2='RandomForest')
        best_feature = df_group.columns[0]
        important_features.append(best_feature)
    # print(important_features)
    features_to_discard = set(corr_features) - set(important_features)
    features_to_discard = list(features_to_discard)
    if showSelection == True:
        print('Features to Discard:', features_to_discard)
        print('Correlating Features to Keep:', important_features)
    return features_to_discard
def getBestFeature(df, method1, method2='RandomForest'):
    '''Sort features by input ranking method.
        method1: method1 = 'mig' , 'auc' , 'modelBased' , 'treeBased'
        method2: If the method uses a machine learning method, this is necessary:
             'DecisionTree', 'RandomForest', 'ExtremeTree', 'XGB' '''
    if method1 == 'mig':
        df_return = mutual_information_gain_FS(df,threshold=0)[0]
    elif method1 == 'auc':
        df_return = ROC_AUC(df, threshold=0, ML=method2, auc_mod=True)
    elif method1 == 'modelBased':
        df_return = model_based_ranking(df, threshold=0, ML=method2)
    elif method1=='treeBased':
        df_return = tree_based_FS(df,k=1,ML=method2,importance_method='MDI')
    return df_return

def multi_correlation_V2(df, threshold=0.85, corr_method='pearson', showGroups=False, showElim=False):
    '''Remove correlating features.
    Input:
        df: Dataframe with features and damage classes
        threshold: features with correlation coefficient higher than this will be grouped
        corr_method: Correlation method. Choose between 'pearson', 'spearman'
        showGroups: Display the groups of correlating features'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    corrmat = X.corr(method=corr_method)
    # Create list with features that have correlation higher than threshold with at least one other feature
    corr_features = find_correlated_features(corrmat,threshold)
    # Create groups of features which correlate
    correlated_groups_list=corr_grouping(corrmat,threshold,showGroups=showGroups)
    # Keep the best feature in each group
    corr_features_to_discard = corr_group_feature_importance(correlated_groups_list,corr_features,X,y,showSelection=False)
    if showElim == True:
        print('Eliminated by Correlation Method: %i' %len(corr_features_to_discard))
        print(corr_features_to_discard)
    X = X.drop(corr_features_to_discard, axis=1)
    X['Damage'] = y
    return X, corr_features_to_discard

# ----------------Wrapper Methods-----------------

def wrapper_FS(df, forward=True, method='DecisionTree', keep_k=0, floating=False, showElim=False):
    '''Wrapper methods.
        Input:
            df: Dataframe with features and damage classes
            forward: True / False. Select between forward or backwards wrapper method
            ML: Machine Learning method to use. Select between 'LogReg', 'DecisionTree',
                                    'RandomForest', 'ExtremeTree', 'XGB'
            keep_k: select maximum features to keep. Stops when k is reached.
                If k=0, the whole feature space is checked and the best subset is selected.
            floating: True/False. Select if floating is enabled or not.'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    clf = ML_method(method)
    if keep_k == 0 or keep_k>len(X.columns):
        keep_k = len(X.columns)
    wrapper = SFS(clf, k_features=(1, keep_k), forward=forward, floating=floating, scoring='accuracy', cv=4, n_jobs=1)
    wrapper = wrapper.fit(X, y)
    df_info = pd.DataFrame.from_dict(wrapper.get_metric_dict()).T
    feature_sets = list(df_info.feature_names)
    score = list(df_info.avg_score)
    best_score = wrapper.k_score_
    best_feature_set = wrapper.k_feature_names_
    feature_order = []
    if forward:
        i_lim = 0
        step = 'Forward'
    else:
        i_lim = keep_k - 1
        step = 'Backward'
    for i in range(keep_k):
        if i == i_lim:
            feature_order.append(''.join(set(feature_sets[i])))
            continue
        if forward == True:
            j = i - 1
        else:
            j = i + 1
        feature_order.append(''.join(set(feature_sets[i]).difference(feature_sets[j])))
    num_features = score.index(best_score)+1
    lastFeature = feature_order[num_features-1]
    score = pd.Series(score,index=feature_order)
    keep = score[0:num_features].index
    eliminate = score[num_features:].index
    if forward == False:
        keep, eliminate = eliminate, keep
    if keep_k != len(X.columns):
        ignored_features = list(set(X.columns).symmetric_difference(set(feature_order)))
        print('Ignored Features (Wrapper):')
        print(ignored_features)
    if showElim == True:
        print('Features Eliminated from %s Feature Selection: %i'%(step,eliminate))
        print(eliminate)
    X = X[keep]
    X['Damage'] = y
    return X

# ----------------Embedded Methods-----------------

def regularization_FS(df, regu_method='L1', showElim=False):
    '''Regularization methods.
    Can change internal variables in methods. Features eliminated sensitive to C.
    See LogisticRegression documentation for explanations
        Input:
            df: Dataframe with features and damage classes
            regu_method: Select regularization method. Select between 'L1', 'L2', 'ElasticNet' '''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    if regu_method=='L1':
        clf = LogisticRegression(penalty='l1',C=0.05,solver='saga',multi_class='ovr')
    elif regu_method == 'L2':
        clf = LogisticRegression(penalty='l2',C=1,solver='saga',multi_class='ovr')    #class_weight='balanced'
    elif regu_method == 'ElasticNet':
        clf = LogisticRegression(penalty='elasticnet',C=1,solver='saga', l1_ratio=0.5)
    sel = SelectFromModel(clf)
    sel.fit(X, y)
    coef_score = (np.mean(abs(sel.estimator_.coef_),axis=0))
    coef_score = pd.Series(coef_score,index=X.columns)
    coef_score = coef_score.sort_values(ascending=False)
    num_features = sum(sel.get_support())
    selected_features = coef_score[0:num_features].index
    elim_features = coef_score[num_features:].index
    if showElim == True:
        print('Features eliminated from %s Regularization: %i' % (regu_method,len(elim_features)))
        print(elim_features)
    X = X[selected_features]
    X['Damage'] = y
    return X, coef_score, elim_features

def kNN_importance(df, k=0, showElim=False):
    '''kNN feature importance.
        input:
            df: Dataframe with features and damage classes
            k: number of features to keep. If k = 0, no features are eliminated.'''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    if k == 0 or k > len(X.columns):
        k = len(X.columns)
    model = KNeighborsClassifier()
    model.fit(X, y)
    results = permutation_importance(model, X, y, n_repeats=10,n_jobs=-1, scoring='accuracy') #
    importance = results.importances_mean
    # Creating a dataframe with the results
    importance = pd.Series(importance,index = X.columns)
    importance = importance.sort_values(ascending=False)
    selected_features = importance[0:k].index
    elim_features = importance[k:].index
    if showElim == True:
        print('Features eliminated from kNN importance, k = %i:' %k)
        print(elim_features)
    X = X[selected_features]
    X['Damage'] = y
    return X, importance

def tree_based_FS(df, k=0, ML='RandomForest', importance_method='MDA', showElim=False):
    '''Tree-based feature importance.
        input:
            df: Dataframe with features and damage classes
            k: number of features to keep. If k = 0, no features are eliminated.
            ML: Machine Learning method to use. Select between 'DecisionTree', 'RandomForest',
                            'ExtremeTree', 'XGB'
            importance_method: Select between 'MDI', 'MDA'. See documentation for more explanation. '''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    clf = ML_method(ML)
    if clf == 'XGB':
        clf = XGBClassifier(reg_alpha=1, reg_lambda=0)
    clf.fit(X, y)
    if ML == 'LogReg':
        importance = clf.coef_[0]
        importance_method = 'MDI'
    else:
        if importance_method == 'MDI':
            importance = clf.feature_importances_
        elif importance_method == 'MDA':
            importance_MDA = permutation_importance(clf, X, y, n_repeats=10)
            sorted_idx = importance_MDA.importances_mean.argsort()
            importance = importance_MDA.importances_mean
    importance = pd.Series(importance, index=X.columns)
    importance = importance.sort_values(ascending=False)
    if k == 0 or k > len(X.columns):
        k = len(X.columns)
    selected_features = importance[0:k].index
    elim_features = importance[k:].index
    if showElim == True:
        print('Features eliminated from %s importance, k = %i:' %(ML,k) )
        print(elim_features)
    X = X[selected_features]
    X['Damage'] = y
    return X, importance, elim_features

def boruta_FS(df, ML='ExtremeTree', showElim=True):
    '''Boruta feature importance.
        input:
            df: Dataframe with features and damage classes
            ML: Machine Learning method to use. Select between 'RandomForest', 'ExtremeTree', 'XGB' '''
    X = np.array(df.drop(['Damage'], 1))
    y = np.array(df.Damage)
    clf = ML_method(ML)  # RandomForest, ExtremeTree, XGB
    feature_selector = BorutaPy(clf, n_estimators='auto', verbose=2)    # verbose = 2 to see progress
    feature_selector.fit(X, y)
    X_filtered = feature_selector.transform(X)
    feature_names = df.drop(['Damage'], 1).columns
    feature_ranks = list(zip(feature_names, feature_selector.ranking_, feature_selector.support_))
    col_names = ['Feature', 'Ranking', 'Keep']
    boruta_df = pd.DataFrame(feature_ranks, columns=col_names)
    boruta_df = boruta_df.sort_values(by=['Ranking'],ignore_index=True)
    print(boruta_df)
    ranking = pd.Series(feature_selector.ranking_, feature_names)
    ranking = ranking.sort_values(ascending=True)
    keep = []
    eliminate = []
    for index, element in enumerate(boruta_df.Keep):
        if element == True:
            keep.append(boruta_df.Feature[index])
        else:
            eliminate.append(boruta_df.Feature[index])
    if showElim == True:
        print('Eliminated by Boruta: %i'%len(eliminate))
        print(eliminate)
    X = pd.DataFrame(X_filtered, columns=keep)
    X['Damage'] = df.Damage
    return X, ranking, eliminate, feature_selector

# ----------------Hybrid Methods-----------------

def recursive_feature_elimination(df, cv=True, rankLim = 0, method='RandomForest', showElim=False):
    # ML: LogReg, DecisionTree, RandomForest, ExtremeTree, XGB
    '''Recursive Feature Selection with our without cross validation.
        Returns the optimal feature space and scores these features as 1 and
        the rest of the features in ascending order depending on their performance.
        input:
            df: Dataframe with features and damage classes
            cv: bool. Include cross validation. Is set to 10 if included.
            rankLim: select how many features to include. If 0, all features are included.
            method: Machine Learning method to use. Select between 'LogReg', 'DecisionTree',
                                    'RandomForest', 'ExtremeTree', 'XGB' '''
    X = df.drop(['Damage'], 1)
    y = df.Damage
    clf=ML_method(method)
    if cv == False:
        recursive = RFE(clf, n_features_to_select=1)
        titleText = ''
    else:
        recursive = RFECV(clf, min_features_to_select=1,cv=10,step=1)
        titleText = 'w. Cross Validation'
    recursive = recursive.fit(X, y)
    names = X.columns
    ranking = pd.Series(recursive.ranking_, index=names)
    ranking = ranking.sort_values(ascending=True)
    if rankLim == 0:
        rankLim = max(ranking)
    keep = ranking.index[ranking<=rankLim]
    eliminate = ranking.index[ranking>rankLim]
    if showElim == True:
        print('Features eliminated with RFE: %i'%eliminate)
        print(eliminate)
    X = X[keep]
    X['Damage'] = y
    return X

# ----------Dimensional Reduction Methods------------
def Dimensional_Reduction(method='PCA', components=5, data=[]):
    '''Dimensional reduction methods using PCA or LDA. 
        Creates the components of the chosen method and returns the explained variance
        of all the components and the first and second component. 
        input: 
            method: 'PCA' or 'LDA'
            components: Number of components to calculate
            df: Dataframe with features and damage classes'''
    X = data.drop(['Damage'], 1)
    y = data.Damage
    if method == 'PCA':
        model = PCA(n_components=components)
        model_fit = model.fit(X)
    else:
        model = LinearDiscriminantAnalysis(n_components=components)
        model_fit = model.fit(X,y)
    Explained_variance = model_fit.explained_variance_ratio_
    Component_1 = model_fit.transform(X)[:,0]
    Component_2 = model_fit.transform(X)[:,1]
    return Explained_variance, Component_1, Component_2




#------- EXAMPLE OF SETUP -----------

# Define splits
splits = 50

# Get features
# df = pd.read_pickle('n_splits_50_sensor_AL01.pkl')
features = getFeatureList(splits, glob=True)
# if selected global:
df = pd.read_pickle('Freq_output_50_all.pkl')

# Include multiple sensors:
df = combineSensors(getSensorList('l'), splits, features)

    # If using only a single sensor:
# df = pd.read_pickle('Features/Features_All/n_splits_50_sensor_AL02.pkl')
# df = chooseFeatures(df,wantedFeatures)

print('Original Feature Space Size: %i'%(len(df.columns)-1))

# Remove damage states (last interger is damage state to keep)
# df = df.loc[df.Damage.isin([0,2])].reset_index(drop=True)

    # Preprocessing (NB! Use after constant & Quasi constant if this is included!)
df = df_preprossecing(df)

# FEATURE SELECTION FUNCTIONS


