from matplotlib.pyplot import imshow, show
from roads import roads
from cv2 import imread

inputFoto = imread("26728705_15.tiff")

final = roads(inputFoto)

imshow(final)
show()
