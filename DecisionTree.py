import numpy as np

# data
# 'Machine Learning' Zhihua Zhou
# P80   Chart 4.2
data = np.array(
    [
        [0,0,0,0,0,0,0],
        [1,0,1,0,0,0,0],
        [1,0,0,0,0,0,0],
        [0,1,0,0,1,1,0],
        [1,1,0,1,1,1,0],
        [0,0,1,0,0,0,0],
        [2,0,0,0,0,0,0],
        [1,1,0,0,1,0,0],
        [0,2,2,0,2,1,1],
        [2,1,1,1,0,0,1],
        [1,1,0,0,1,1,1],
        [2,0,0,2,2,0,1],
        [0,0,1,1,1,0,1],
        [1,1,1,1,1,0,1],
        [2,2,2,2,2,0,1],
        [2,0,0,2,2,1,1],
        [0,1,0,1,0,0,1]
    ]
)

x = data[:,:-1]
y = data[:,-1]



# function
def nodeEntropy(para_y):
    labelList = para_y.tolist()
    LabelSet = set(labelList)
    totalNum = para_y.shape[0]
    pVector = np.array([labelList.count(c)/totalNum for c in LabelSet])
    return -1*np.sum([pVector*np.log2(pVector)], axis=1)
    

def featureGain(para_data, para_rootEntropy, para_featureIndex, para_discrete=True):
    x = para_data[:,:-1]
    y = para_data[:,-1]
    labelList = y.tolist()
    LabelSet = set(labelList)
    LabelNum = len(LabelSet)
    if para_discrete:
        featureList = x[:,para_featureIndex].tolist()
        featureSet = set(featureList)
        featureNum = len(featureSet)
        # record subset index
        # feature {0,1} 
        # countDic = {
        #   0:[0,3,5]
        #   1:[1,2,4]
        # }
        countDict = dict(zip(featureSet, [[] for __ in range(featureNum)]))
        for i in range(len(featureList)):
            countDict[featureList[i]].append(i)
    else:
        # TODO handle continue feature
        pass
    subsetDict = dict()
    for k,v in countDict.items():
        subsetDict[k] = np.array([para_data[i] for i in v])
    sumEntropy = sum([
        nodeEntropy(v[:,-1])*v.shape[0]/data.shape[0]
        for v in subsetDict.values()
    ])
    return para_rootEntropy - sumEntropy


def selectFeatureSplit_ID3(para_data, splitFeatureList):
    """
        Split feature with information gain
    """
    x = para_data[:,:-1]
    y = para_data[:,-1]
    rootEntropy = nodeEntropy(y)
    featureGainList = [featureGain(para_data, rootEntropy, i) for i in splitFeatureList]
    return featureGainList.index(max(featureGainList))


def trainDecsionTree(para_data, method=selectFeatureSplit_ID3):
    return None
    pass


featureList = [i for i in range(x.shape[1])]
