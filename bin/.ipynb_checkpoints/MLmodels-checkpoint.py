import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import learning_curve



def evaluation(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, scoring='f1', train_sizes=np.linspace(0.1, 1, 10))
    
    fig = plt.figure(figsize=(6,4))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    #display(fig)
    plt.legend()
    
    plt.show()
    
    
    
    
    
    
    
def pipeline(peakTable):
      
    print(180*'-')
    print(20*'-', 'peakTable', 140*'-')
    print(180*'-')

    # Split into train and test sets
    code = {'Patient':1, 'Control':0}
    
    peakTable_Sample = peakTable[peakTable['SampleType'] == 'Sample']
    
    y = peakTable_Sample['Class']
    y = y.map(code)
    X = peakTable_Sample.iloc[:, [col[0] == 'M' for col in peakTable_Sample.columns]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2, random_state=0, stratify=y)

    # Create and evaluate models
    prepocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))

    RandomForest = make_pipeline(prepocessor, RandomForestClassifier(random_state=0))
    AdaBoost = make_pipeline(prepocessor, AdaBoostClassifier(random_state=0))
    SVM = make_pipeline(prepocessor, StandardScaler(), SVC(random_state=0))
    KNN = make_pipeline(prepocessor, StandardScaler(), KNeighborsClassifier(n_neighbors = 3))

    dict_models = {'RandomForest': RandomForest,
           'AdaBoost' : AdaBoost,
           'SVM': SVM,
           'KNN': KNN
          }

    for name, model in dict_models.items():
        print(name)
        evaluation(model, X_train, X_test, y_train, y_test)
        print(100*'-')


    print(3*'\n')