import numpy as np
from utils import snr,corr


def PoiSelection(x_train, y_train,x_test, y_test, poi_type='corr', poi_num = 10, poi_idx=[]):
    
    # PoI selection module，select poi_num POIs according to the highest SNR and Pearson's correlation coefficent. Or just feed it with your own POI list.
    # Example:
    # X_profiling_poi = PoiSelection(X_profiling,poi_type='corr',poi_num=20)
    
    if poi_type == 'corr':
        m = -np.abs(corr(x_train,y_train))
        poi_idx = np.argsort(m,axis=0)[:poi_num]
    elif poi_type == 'snr':
        m = -snr(x_train,y_train)
        poi_idx = np.argsort(m,axis=0)[:poi_num]
    elif poi_type == 'custom':
        pass
    else:
        print('Error:unknown poi type')
        
    x_train = x_train[:,poi_idx]
    x_test = x_test[:,poi_idx]
    return x_train,x_test