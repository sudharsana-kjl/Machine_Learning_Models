import loadDataset
import KNearestNeighbor
	
trainingSet=[]
testSet=[]
split = 0.67

loadDataset.loadDataset('iris.data', split, trainingSet, testSet)
print 'Train set: ' + repr(len(trainingSet))
print 'Test set: ' + repr(len(testSet))
# generate predictions
predictions=[]
k = 3
for x in range(len(testSet)):
	neighbors = KNearestNeighbor.getNeighbors(trainingSet, testSet[x], k)
	result = KNearestNeighbor.getResponse(neighbors)
	predictions.append(result)
	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = KNearestNeighbor.getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
	

