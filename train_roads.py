# -*- coding: utf-8 -*-'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from os.path import isfile
from keras.models import load_model
import numpy as np
import cv2
import os

FOLDERPHOTOS = "mass_roads/train/sat"
FOLDERETIQUETES = "mass_roads/train/map"

inputFoto = cv2.imread("10078660_15.tiff")
outputFoto = cv2.imread("10078660_15.tif")
part = inputFoto[0:5,0:5,:]
num_classes=2

img_rows = 5
img_cols = 5
colors = 3
orig_rows, orig_cols, orig_colors = inputFoto.shape
input_shape = (img_rows, img_cols, colors)

if isfile("model.h5"):
	model = load_model("model.h5")
else:
	model = Sequential()
	model.add(Conv2D(4, kernel_size=(3, 3),             # 4 filtry 3x3
                 activation='relu',                  
                 input_shape=input_shape))

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())                                 # przed warstwą Dense musimy "spłaszczyć" dane [height,width,channels]->[length]
	#model.add(Dense(128, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))  # softmax: suma wyjść równa 1 (interpretujemy jako prawdopodobieństwo)

	model.compile(loss=keras.losses.categorical_crossentropy,  # odpowiednia funkcja straty
              optimizer=keras.optimizers.Adadelta(),       # wybrany optymalizator
              metrics=['accuracy'])      

model.summary()

batch_size = 50   # ile obrazków przetwarzamy na raz (aktualizacja wag sieci następuje raz na całą grupę obrazków)
epochs = 1         # ile epok będziemy uczyli

photosList = os.listdir(FOLDERPHOTOS)

counter2 = 0

for p in photosList:
	if str(p).endswith(".tiff"):
		counter2+=1
		x_train = np.zeros(((orig_rows-4)*(orig_cols-4), 5, 5, 3), dtype=np.object_)
		y_train = np.zeros((orig_rows-4)*(orig_cols-4))
		x_train_positives = []
		y_train_positives = []
		counter=0
		inputFoto = cv2.imread(FOLDERPHOTOS + "/" + p)
		outputFoto = cv2.imread(FOLDERETIQUETES + "/" + p[:-1])

		#print("x_train last " + str(x_train[len(x_train) -1]))
		#print("photo " + str(p))
		for i in range (2, orig_rows-2):
			for j in range(2, orig_cols - 2):
				fragment = inputFoto[i-2:i+3,j-2:j+3,:]
				#print("fragment " + str(fragment))
				x_train[counter] = fragment
				if outputFoto[i][j][0] > 0 or outputFoto[i][j][1] > 0 or outputFoto[i][j][2] > 0:
					#print("i j >0 " + str(i) + " " + str(j))
					y_train[counter]=1
					#x_train = np.append(x_train, np.tile(fragment, 10), axis=0)
					#y_train = np.append(y_train, np.tile(1, 10), axis=0)
					#print("x " + str(x_train_positives))
					#x_train_positives = x_train_positives.append(np.tile(fragment, 10))
					#y_train_positives = y_train_positives.append(np.tile(1, 10))
					for k in range(15):
						#print("k " + str(k))
						#x_train = np.append(x_train, [fragment], axis=0)
						#y_train = np.append(y_train, [1], axis=0)
						x_train_positives.append(fragment)
						y_train_positives.append(1)
					#print("x_train last " + str(x_train[len(x_train) -1]))
				#else:
				#	y_train[counter]=0 #niepotrzebne bo tablica zainicjalizowana zerami
				counter+=1
		#x_train = x_train.astype('float32')
		#x_train /= 255

		#print("after for " + str(p))

		x_train = np.append(x_train, x_train_positives, axis=0)
		y_train = np.append(y_train, y_train_positives, axis=0)

		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, colors)
		#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

		y_train = keras.utils.to_categorical(y_train, num_classes)

		#print("before fit " + str(p))

		model.fit(x_train, y_train,
          	batch_size=batch_size,
          	epochs=epochs,
          	verbose=1)

		model.save("model_" + str(counter2) + ".h5")
		if counter2 > batch_size:
			break

model.save("model.h5")
