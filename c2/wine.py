import numpy as np
import csv
import operator

def generateData(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = [row for row in reader]
        data = data[1:]

    data = np.array(data)

    label = data[:,-1]
    data = data[:,:-1]

    data = data.astype(np.float)
    label = label.astype(np.float)

    #normlize to [0,1]
    minVals = data.min(0)
    maxVals = data.max(0)
    data = (data-minVals)/(maxVals-minVals)

    return data, label

def knnclassify(testData, trainData, trainLabel,k):
    diff = trainData - testData
    diffSq = diff**2
    distances = np.sqrt(np.sum(diffSq, axis = 1))
    sortedDistdicies = distances.argsort()
    classCount = {}
    for i in range(k):
        votedLabel = trainLabel[sortedDistdicies[i]]
        classCount[votedLabel] = classCount.get(votedLabel, 0) + 1
    #print len(classCount)
    #print classCount
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


testData = np.array([0.2,0.3,0.11,0.13,0.16,0.2,0.3,0.11,0.13,0.16,0.27])
data, label = generateData('winequality-red.csv')
#print label[3]
k=9
level = knnclassify(testData, data, label, k)
print level


data, label = generateData('winequality-red.csv')
ratio = 0.1
k = 9
m = int(ratio*data.shape[0])
testData = data[:m];
trainData = data[m:]
testLabel = label[:m]
trainLabel = label[m:]
print testData.shape, trainData.shape, testLabel.shape, trainLabel.shape

errorCount = 0
for i in range(testData.shape[0]):
    predict = knnclassify(testData[i], trainData, trainLabel, k)
    print "the predict is %f, the ground truth is %f" %(predict, testLabel[i])
    if(predict != testLabel[i]):
        errorCount = errorCount + 1;

print "the total error rate is %f" %(errorCount/float(testData.shape[0]))
