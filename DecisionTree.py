import numpy as np

data = np.array(
    [
        [0,0,0,1],
        [0,0,1,1],
        [1,0,0,1],
        [0,1,0,1],
        [1,1,0,0],
        [1,1,1,0],
        [0,1,1,0],
        [1,0,1,0]
    ]
)
x = data[:,:3]
y = data[:,3]

def nodeEntropy(para_y):
    yList = para_y.tolist()
    classSet = set(yList)
    classNum = len(classSet)
    totalNum = para_y.shape[0]
    # return -1*sum([p*np.log(p) for p in [yList.count(c)/totalNum for c in classSet]])
    pVector = np.array([yList.count(c)/totalNum for c in classSet])
    return -1*np.sum([pVector*np.log(pVector)], axis=1)


def featureGain(para_x, para_y, para_feature_index, para_discrete=True):
    yList = para_y.tolist()
    classSet = set(yList)
    classNum = len(classSet)
    rootEntropy = nodeEntropy(para_y)
    if para_discrete:
        xList = para_x[para_feature_index,:].tolist()
        xValueSet = set(xList)
        xValueNum = len(xValueSet)
        # x=0,1,2 y=0,1
        # x_dict = {
        #   0:{0:0, 1:0}
        #   1:{0:0, 1:0}
        #   2:{0:0, 1:0}
        # }  
        # think how to generate dict with counting at the same time
        x_dict = zip(xValueSet, [zip(classSet, [0 for __ in range(classNum)]) for __ in range(xValueNum)])
        for i in range(len(xList)):
            pass
    else:
        pass
    return None


def selectFeatureSplit(para_x, para_y):
    """
        Split with feature para_fea
        para_fea:   the index of feature
    """
    entropy_root = nodeEntropy(para_y)
    # label 
    yList = para_y.tolist()
    classSet = set(yList)
    classNum = len(classSet)    
    # feature
    featureNum = para_x.shape[1]
    for i in range(featureNum):
        # TODO calculate gain of each feature
        pass



    return None