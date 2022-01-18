
'''
imputes missing values in peak table with constant value given as parameters

inputs :
    - X : peakTable with only variable columns, no metadata
    - const (default=0): value to impute in place of NaN
'''
def const_imputer(X, const=0):
    
    X_const = X.copy()
    X_const[X_const.isna()] = const
    return X_const



'''
For each feature, the missing values are imputed by the mean value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def mean_imputer(X):
    
    import pandas as pd
    from sklearn.impute import SimpleImputer
    
    imp = SimpleImputer(strategy='mean')
    imp.fit(X)

    X_mean = pd.DataFrame(imp.transform(X), columns=X.columns)
    return X_mean


'''
For each feature, the missing values are imputed by the median value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def median_imputer(X):

    import pandas as pd
    from sklearn.impute import SimpleImputer
    
    imp = SimpleImputer(strategy='median')
    imp.fit(X)

    X_median = pd.DataFrame(imp.transform(X), columns=X.columns)
    return X_median


'''
For each feature, the missing values are imputed by the most frequent value (rounded at 1.e-2) of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def mode_imputer(X):

    import pandas as pd
    from sklearn.impute import SimpleImputer
    
    imp = SimpleImputer(strategy='most_frequent')
    imp.fit(round(X,2))

    X_most = pd.DataFrame(imp.transform(X), columns=X.columns)
    return X_most



'''
For each feature, the missing values are imputed by the minimum value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def min_imputer(X):

    X_min = X.fillna(value=X.min())
    return X_min


'''
For each feature, the missing values are imputed by the half of the minimum value of the non-missing values in that feature

input :
    - X : peakTable with only variable columns, no metadata
'''
def half_min_imputer(X):

    X_half_min = X.fillna(value=X.min()/2)
    return X_half_min




'''
For each feature, the missing values are imputed using the MICE method (Multivariate Imputation by Chained Equations), inspired by the R MICE package.

input :
    - X : peakTable with only variable columns, no metadata
'''
def python_MICE_imputer(X, estimator):
    
    import pandas as pd
    
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    from sklearn.linear_model import BayesianRidge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor

    import time
    start_time = time.time()

    imp = IterativeImputer(estimator=estimator, max_iter=10, random_state=0, n_nearest_features=10)
    imp.fit(X)

    X_python_MICE = pd.DataFrame(imp.transform(X), columns=X.columns)
    
    print("----- {0:.1f} seconds -----".format(time.time() - start_time))
          
    return X_python_MICE




'''
"Each missing feature is imputed using values from n_neighbors nearest neighbors that have a value for the feature. The feature of the neighbors are averaged uniformly or weighted by distance to each neighbor. If a sample has more than one feature missing, then the neighbors for that sample can be different depending on the particular feature being imputed."

inputs :
    - X : peakTable with only variable columns, no metadata
    - n_neighbors (default=5) : number of neighbors to impute each feature
    - by (default='features') : allows to choose axis along which perform the imputation
'''
def KNN_imputer(X, n_neighbors=5, by='features'):
    
    import pandas as pd
    from sklearn.impute import KNNImputer
    
    imp = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    
    if by=='samples':

        imp.fit(X)

        X_imp_KNN = pd.DataFrame(imp.transform(X), columns=X.columns)
        return X_imp_KNN
    
    elif by=='features':
        
        imp.fit(X.transpose())

        X_imp_KNN = pd.DataFrame(imp.transform(X.transpose()), columns=X.transpose().columns).transpose()
        return X_imp_KNN
    
    else:
        print('Wrong argument for <by>')


