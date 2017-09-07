import cPickle
import gzip
import numpy as np

def load_data():
	f = gzip.open('mnist.pkl.gz','rb')
	train_data,valida_data,test_data = cPickle.load(f)
	f.close()
	return (train_data,valida_data,test_data)

def load_data_wrapper():
	train_data, valida_data, tes_data = load_data()
   	training_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    	training_results = [vectorized_result(y) for y in train_data[1]]
    	training_data = zip(training_inputs, training_results)
    	validation_inputs = [np.reshape(x, (784, 1)) for x in valida_data[0]]
    	validation_data = zip(validation_inputs, valida_data[1])
    	test_inputs = [np.reshape(x, (784, 1)) for x in tes_data[0]]
    	test_data = zip(test_inputs, tes_data[1])
    	return (training_data, validation_data, test_data)

def vectorized_result(j):
	e = np.zeros((10,1))
	e[j] = 1.0
	return e
