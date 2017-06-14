from keras.models import load_model
from skimage.filters import median
from skimage.morphology import disk
from numpy import zeros, empty

def roads(image):
	model = load_model("model.h5")
	img_rows = 5
	img_cols = 5
	colors = 3

	orig_rows, orig_cols, orig_colors = image.shape
	input_shape = (img_rows, img_cols, colors)

	result = zeros((orig_rows, orig_cols))

	for i in range (2, orig_rows - 2):
		for j in range(2, orig_cols - 2):
			temp = empty((1, img_rows, img_cols, colors))
			temp[0] = image[i-2:i+3,j-2:j+3,:]

			prediction = model.predict(temp, batch_size=1, verbose=1)
			if prediction[0][1] > prediction[0][0]:
				result[i][j]=1

	return median(result, disk(5))
