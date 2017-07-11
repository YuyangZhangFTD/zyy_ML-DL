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
x = data[:,:-1]
y = data[:,-1]
# ==================================================================================


# ================================== function ====================================== 
def nodeEntropy(para_y):
    labelList = para_y.tolist()
    labelSet = set(labelList)
    totalNum = para_y.shape[0]
    pVector = np.array([labelList.count(c)/totalNum for c in labelSet])
    return -1*np.sum([pVector*np.log2(pVector)], axis=1)
    

def featureGain(para_data, para_rootEntropy, para_featureIndex, para_discrete=True):
    x = para_data[:,:-1]
    y = para_data[:,-1]
    labelList = y.tolist()
    labelSet = set(labelList)
    labelNum = len(labelSet)
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


def selectFeatureSplit_ID3(para_data, para_splitFeatureList):
    """
        Split feature with information gain
    """
    x = para_data[:,:-1]
    y = para_data[:,-1]
    rootEntropy = nodeEntropy(y)
    featureGainList = [featureGain(para_data, rootEntropy, i) for i in para_splitFeatureList]
    return featureGainList.index(max(featureGainList))


def stopGrowing(para_data, para_splitFeatureList, featureDict, splitDataDict):
    """
        1. all nodes belong to same label
        2. split feature set is empty or the value of subset of data are same in feature A 
        3. subset of data is empty
    """
    pass
    return None


def trainDecsionTree(para_data, para_splitFeatureList, para_treeDict, featureDict, method=selectFeatureSplit_ID3):
    x = para_data[:,:-1]
    y = para_data[:,-1]
    
    # stop growing
    if len(y)==0 or len(para_splitFeatureList)==0 or y[0]==np.mean(y):
        return para_treeDict

    # select feature to split
    splitNode = selectFeatureSplit_ID3(para_data, para_splitFeatureList)

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
        trainDecsionTree(v, copy.copy(para_splitFeatureList), para_treeDict[(splitNode, k)], featureDict, method)
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
            5:(0,1)
        }
featureList = list(featureDict.keys())
treeDict = {}
trainDecsionTree(data, featureList, treeDict, featureDict, method=selectFeatureSplit_ID3)
print(treeDict)
# ==================================================================================