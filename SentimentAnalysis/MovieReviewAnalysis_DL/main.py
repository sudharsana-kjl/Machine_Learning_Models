#A one dimensional CNN model for IMDB dataset

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

seed=7 
numpy.random.seed(seed) #fixing random seed for reproducibilty

"""top_words = 5000
(Xtrain,Ytrain),(Xtest,Ytest) = imdb.load_data(num_words=top_words) #loading dataset but keeping only top n words and zero the remaining
max_words = 500
Xtrain = sequence.pad_sequences(Xtrain,maxlen=max_words) #pad dataset to maximum review length in words
Xtest = sequence.pad_sequences(Xtest,maxlen=max_words)

model = Sequential()
model.add(Embedding(top_words,32,input_length=max_words))
model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(Xtrain,Ytrain,validation_data=(Xtest,Ytest),epochs=2,batch_size=128,verbose=2)
scores = model.evaluate(Xtest,Ytest,verboe=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
