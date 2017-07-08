import numpy as np

# data
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



# function
def nodeEntropy(para_y):
    yList = para_y.tolist()
    classSet = set(yList)
    classNum = len(classSet)
    totalNum = para_y.shape[0]
    # return -1*sum([p*np.log(p) for p in [yList.count(c)/totalNum for c in classSet]])
    pVector = np.array([yList.count(c)/totalNum for c in classSet])
    return -1*np.sum([pVector*np.log(pVector)], axis=1)


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
        pass

    subsetDict = dict()
    for k,v in countDict.items():
        subsetDict[k] = np.array([para_data[i] for i in v])
    sumEntropy = sum([
        nodeEntropy(v[:,-1])*v.shape[0]/data.shape[0]
        for v in subsetDict.values()
    ])
    return para_rootEntropy - sumEntropy


def selectFeatureSplit(para_x, para_y, splitFeatureList, splitFeatureDict):
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


# draft
# def featureGain(para_x, para_y, para_rootEntropy, para_featureIndex, para_discrete=True)
#     labelList = para_y.tolist()
#     LabelSet = set(labelList)
#     LabelNum = len(LabelSet)
#     if para_discrete:
#         featureList = para_x[:,para_featureIndex].tolist()
#         featureSet = set(xList)
#         featureNum = len(featureSet)
#         # x=0,1,2 y=0,1
#         # x_dict = {
#         #   0:{0:0, 1:0}
#         #   1:{0:0, 1:0}
#         #   2:{0:0, 1:0}
#         # }  
#         # think how to generate dict with counting at the same time
#         countDict = dict(zip(
#                 featureSet, [
#                     dict(zip(LabelSet, [ 0 for __ in range(LabelNum)]))
#                     for __ in range(featureNum)
#                 ]))
#         for i in range(len(featureList)):
#             countDict[featureList[i]][labelList[i]] += 1
#     else:
#         pass