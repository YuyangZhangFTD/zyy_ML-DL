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
        for discrete variable, split with number of variable value
        for continue variable, split with number of label value 
    """
    if para_discrete:
        subsetData = dict(zip(
            featureDict[para_feature],
            [
                data[np.where(x[:,para_feature]==k)]
                for k in featureDict[para_feature]
            ]
        ))
    else:
        labelNum = len(set(para_data[:,-1].tolist()))
        nodeNum = int(y.shape[0]/labelNum)
        para_data.sort(para_data[:,para_feature].argsort())
        subsetData = dict(zip(
            [i for i in range(labelNum)],
            [
                data[i*nodeNum:(i+1)*nodeNum]
                if i != (labelNum-1)
                else data[i*nodeNum:]
                for i in range(labelNum)
            ]

        ))
    return subsetData


def featureGain(para_data, para_rootEntropy, para_feature, featureDict, para_discrete=True):
    """
        ID3: use gain to select feature
    """
    subsetDict = splitSubsetData(para_data, para_feature, featureDict, para_discrete)
    sumEntropy = sum([
        nodeEntropy(v[:,-1])*v.shape[0]/data.shape[0]
        for v in subsetDict.values()
    ])
    return para_rootEntropy - sumEntropy


def featureGainRatio(para_data, para_rootEntropy, para_feature, para_discrete=True):
    """
        C4.5: use gain ratio to select feature
    """
    return None


def featureGiniIndex(para_data, para_rootEntropy, para_feature, para_discrete=True):
    """
        CART: use gini index to select feature
    """
    return None


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
    continueVariableList = [k for k,v in featureDict.items() if len(v)==0]
    compareList = [
        compareMethod(para_data, rootEntropy, i, featureDict) 
        if i not in continueVariableList else
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

    # select feature to split
    splitNode = selectFeatureSplit(para_data, para_splitFeatureList, featureDict, compareMethod)

    # split feature and init tree dict which is used for recording
    # meanwhile, split data
    subsetData = {}
    # if variable is discrete, split number decided by number of feature values
    splitNum = len(featureDict[splitNode])
    if splitNum != 0:
        # del it from splitFeatureList
        para_splitFeatureList.pop(splitNode)
        # feature a = {0,1,2}, split into 3 pieces
        for k in featureDict[splitNode]:
            para_treeDict[(splitNode, k)] = {}
            # numpy is too powerful!!! OTZ
            subsetData[k] = para_data[np.where(x[:,splitNode]==k)]
    else:
        # split number decided by values of label
        # 2-class classification, split into 2 pieces
        splitNum = len(set(y.tolist()))
        # TODO how to record in tree dict
        # do with para_treeDict and subsetData
        print("Unlucky to be here !")

    # recursion 
    for k,v in subsetData.items():
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
# () for continue variable
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
trainDecsionTree(data, featureList, treeDict, featureDict, compareMethod=featureGain)
print(treeDict)
# ==================================================================================
