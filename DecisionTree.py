import numpy as np
import copy

# ================================== init data =====================================
# 'Machine Learning' Zhihua Zhou
# P80   Chart 4.2
data = np.array(
    [
        [0,0,0,0,0,0,0],
        [1,0,1,0,0,0,0],
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [1,1,0,1,0,0,0],
        [0,0,1,0,1,1,0],
        [2,0,0,0,1,1,0],
        [1,1,0,0,1,0,0],
        [0,2,2,0,1,0,1],
        [2,1,1,1,2,1,1],
        [1,1,0,0,2,0,1],
        [2,0,0,2,2,1,1],
        [0,0,1,1,0,0,1],
        [1,1,1,1,0,0,1],
        [2,2,2,2,1,1,1],
        [2,0,0,2,2,0,1],
        [0,1,0,1,1,0,1]
    ]
)
# P84   Chart 4.3
# data = np.array(
#     [
#         [0,0,0,0,0,0,0.697,0.460,0],
#         [1,0,1,0,0,0,0.744,0.376,0],
#         [1,0,0,0,0,0,0.634,0.264,0],
#         [0,1,0,0,1,1,0.608,0.318,0],
#         [1,1,0,1,1,1,0.556,0.215,0],
#         [0,0,1,0,0,0,0.403,0.237,0],
#         [2,0,0,0,0,0,0.481,0.149,0],
#         [1,1,0,0,1,0,0.437,0.211,0],
#         [0,2,2,0,2,1,0.666,0.091,1],
#         [2,1,1,1,0,0,0.243.0.267,1],
#         [1,1,0,0,1,1,0.245,0.057,1],
#         [2,0,0,2,2,0,0.343,0,099,1],
#         [0,0,1,1,1,0,0.639,0.161,1],
#         [1,1,1,1,1,0,0.657,0.198,1],
#         [2,2,2,2,2,0,0.360,0.370,1],
#         [2,0,0,2,2,1,0.593,0.042,1],
#         [0,1,0,1,0,0,0.719,0.103,1]
#     ]
# )
x = data[:,:-1]
y = data[:,-1]
# ==================================================================================


# ================================== function ====================================== 
def nodeEntropy(para_y):
    """
        the entropy of a node
    """
    labelList = para_y.tolist()
    labelSet = set(labelList)
    totalNum = para_y.shape[0]
    pVector = np.array([labelList.count(c)/totalNum for c in labelSet])
    return -1*np.sum([pVector*np.log2(pVector)], axis=1)
    

def splitSubsetData(para_data, para_feature, featureDict, para_discrete=True):
    """
        split data by one feature and record in a dict
        subsetData = {
            featureValue_1: np.array([[...],[...],...,[...]])
            ...
            featureValue_2: np.array([[...],[...],...,[...]])
        }
        for discrete feature, split with number of feature value
        for continuous feature, split with number of label value or just split into 2
    """
    if para_discrete:
        subsetData = dict(zip(
            featureDict[para_feature],
            [
                para_data[np.where(para_data[:,para_feature]==k)]
                for k in featureDict[para_feature]
            ]
        ))
    else:
        # labelNum = len(set(para_data[:,-1].tolist()))
        labelNum = 2    # binary split
        nodeNum = int(y.shape[0]/labelNum)
        para_data.sort(para_data[:,para_feature].argsort())
        subsetData = dict(zip(
            [i for i in range(labelNum)],
            [
                para_data[i*nodeNum:(i+1)*nodeNum]
                if i != (labelNum-1)
                else para_data[i*nodeNum:]
                for i in range(labelNum)
            ]

        ))
    return subsetData


def featureGain(para_data, para_rootEntropy, para_feature, featureDict, para_discrete=True):
    """
        ID3: use gain to select feature
    """
    subsetDict = splitSubsetData(para_data, para_feature, featureDict, para_discrete)
    totalDataNum = sum([v.shape[0] for v in subsetDict.values()])
    sumEntropy = sum([
        nodeEntropy(v[:,-1])*v.shape[0]/totalDataNum
        for v in subsetDict.values()
    ])
    return para_rootEntropy - sumEntropy


def featureGainRatio(para_data, para_rootEntropy, para_feature, featureDict, para_discrete=True):
    """
        C4.5: use gain ratio to select feature
    """
    subsetDict = splitSubsetData(para_data, para_feature, featureDict, para_discrete)
    # just copy from featureGain()
    totalDataNum = sum([v.shape[0] for v in subsetDict.values()])
    gain = para_rootEntropy - sum([
        nodeEntropy(v[:,-1])*v.shape[0]/totalDataNum
        for v in subsetDict.values()
    ])
    valueNum = [v.shape[0] for v in subsetDict.values()]
    valueNumSum = sum(valueNum)
    intrinsicValue = -1 * sum([
        num/valueNumSum * np.log2(num/valueNumSum)
        for num in valueNum
    ])
    return gain/intrinsicValue


def featureGiniIndex(para_data, para_rootEntropy, para_feature, para_discrete=True):
    """
        CART: use gini index to select feature
        para_rootEntropy is not used
    """
    subsetDict = splitSubsetData(para_data, para_feature, featureDict, para_discrete)
    totalDataNum = sum([v.shape[0] for v in subsetDict.values()])
    pVector = [
        v[:,-1].shape[0]/totalDataNum
        for v in subsetDict.values()
    ]
    return 1-sum([p**2 for p in pVector])


def selectFeatureSplit(para_data, para_splitFeatureList, featureDict, compareMethod=featureGain):
    """
        select suitable feature to split data
        compareMethod:
            featureGain         ==>     ID3
            featureGainRatio    ==>     C4.5
            featureGiniIndex    ==>     CART
    """
    x = para_data[:,:-1]
    y = para_data[:,-1]
    rootEntropy = nodeEntropy(y)
    continuousVariableList = [k for k,v in featureDict.items() if len(v)==0]
    compareList = [
        compareMethod(para_data, rootEntropy, i, featureDict) 
        if i not in continuousVariableList else
        compareMethod(para_data, rootEntropy, i, featureDict, False) 
        for i in para_splitFeatureList
    ]
    return compareList.index(max(compareList))


def trainDecsionTree(para_data, para_splitFeatureList, para_treeDict, featureDict, compareMethod=featureGain):
    x = para_data[:,:-1]
    y = para_data[:,-1]
    
    # stop growing
    if len(y)==0 or len(para_splitFeatureList)==0 or y[0]==np.mean(y):
        return para_treeDict
 
    # select feature to split: get faeture value instead of faeture index, so use remove to del discrete feature
    splitNode = para_splitFeatureList[selectFeatureSplit(para_data, para_splitFeatureList, featureDict, compareMethod)]
    
    # split number: judge discrete feature or continuous faeture
    splitNum = len(featureDict[splitNode])
    
    # split feature
    subsetData = splitSubsetData(para_data, splitNode, featureDict, para_discrete=(splitNum != 0))
    # init tree dict which is used for recording tree structure
    if splitNum != 0:
        # del it from splitFeatureList
        para_splitFeatureList.remove(splitNode)
        # if variable is discrete, split number decided by number of feature values
        splitFeatureValueSet = featureDict.get(splitNode)
    else:
        # split number decided by values of label or 2        
        # 0 for samples which is less than middle value
        # 1 for samples which is bigger than middle value
        splitFeatureValueSet = (0,1)

    for k in splitFeatureValueSet:
        para_treeDict[(splitNode, k)] = {}
        
    print("Split Node at:  "+str(splitNode))
    # recursion 
    for k,v in subsetData.items():
        print("Init Node from "+str(splitNode)+" at value: "+str(k))
        trainDecsionTree(
            v, copy.copy(para_splitFeatureList), 
            para_treeDict[(splitNode, k)], featureDict, compareMethod
        )
    return para_treeDict
# ==================================================================================
    

# ================================ test ============================================
# feature dict
# key:      feature name
# value:    feature value
# () for continuous feature
featureDict = {
            0:(0,1,2),
            1:(0,1,2),
            2:(0,1,2),
            3:(0,1,2),
            4:(0,1,2),
            # 6:(),
            # 7:(),
            5:(0,1)
        }
featureList = list(featureDict.keys())
treeDict = {}
trainDecsionTree(data, featureList, treeDict, featureDict, compareMethod=featureGiniIndex)
print(treeDict)
# ==================================================================================
