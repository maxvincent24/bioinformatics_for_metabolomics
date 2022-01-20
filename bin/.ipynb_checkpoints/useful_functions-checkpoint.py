

def remove_features_mv(peakTable, percent_mv_threshold=50):
    
    peakTable_Sample = peakTable[peakTable['SampleType'] == 'Sample']
    X_Sample = peakTable_Sample.iloc[:, [col[0] == 'M' for col in peakTable_Sample.columns]]

    print(f'Number of feature initially in X : {X_Sample.shape[1]}')
    percent_mv = X_Sample.isna().sum() / X_Sample.shape[0] * 100
    
    features = percent_mv[percent_mv > percent_mv_threshold].index
    print(f'Number of features removed : {len(features)}')
    
    peakTable = peakTable.drop(features, axis=1)
    print(f'Number of feature finally in X : {X_Sample.shape[1] - len(features)}')
    
    return peakTable







def remove_features_RSD(peakTable, percent_RSD_threshold):
    
    peakTable_QC = peakTable[peakTable['QC'] == 1]
    X_QC = peakTable_QC.iloc[:, [col[0] == 'M' for col in peakTable_QC.columns]]
    
    print(f'Number of feature initially in X : {X_QC.shape[1]}')
    
    RSD = X_QC.std() / X_QC.mean() * 100
    
    features = RSD[RSD > percent_RSD_threshold].index
    print(f'Number of features removed : {len(features)}')
    
    peakTable = peakTable.drop(features, axis=1)
    print(f'Number of feature finally in X : {X_QC.shape[1] - len(features)}')
    
    return peakTable



