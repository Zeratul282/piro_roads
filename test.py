from keras.models import load_model
from matplotlib.pyplot import imshow, show
import numpy as np
import cv2

def roads(image):
	model = load_model("model.h5")
	img_rows = 5
	img_cols = 5
	colors = 3

	orig_rows, orig_cols, orig_colors = image.shape
	input_shape = (img_rows, img_cols, colors)

	result = np.zeros((orig_rows, orig_cols))

	for i in range (2, orig_rows - 2):
		for j in range(2, orig_cols - 2):
			temp = np.empty((1, img_rows, img_cols, colors))
			temp[0] = inputFoto[i-2:i+3,j-2:j+3,:]

			# temp = temp.reshape(temp.shape[0], img_rows, img_cols, colors)

			prediction = model.predict(temp, batch_size=1, verbose=1)

			# print(prediction)

			if prediction[0][1] > prediction[0][0]:
				result[i][j]=255

	return result

inputFoto = cv2.imread("10078660_15.tiff")

final = roads(inputFoto)

imshow(final)
show()
