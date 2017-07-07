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
    classType = set(yList)
    classNum = len(classType)
    totalNum = para_y.shape[0]
    # return -1*sum([p*np.log(p) for p in [yList.count(c)/totalNum for c in classType]])
    pVector = np.array([yList.count(c)/totalNum for c in classType])
    return -1*np.sum([pVector*np.log(pVector)], axis=1)

def selectFeatureSplit(para_x, para_y):
    """
        Split with feature para_fea
        para_fea:   the index of feature
    """
    entropy_root = nodeEntropy(para_y)
    # label 
    yList = para_y.tolist()
    classType = set(yList)
    classNum = len(classType)    
    # feature
    featureNum = para_x.shape[1]
    for i in range(featureNum):
        # TODO calculate gain of each feature
        pass

    return None