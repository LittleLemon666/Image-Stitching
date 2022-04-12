from PIL import Image
import numpy as np
import os
from os import path

def readFolder(folderPath):

	images = []
	filePaths = os.listdir(folderPath)

	# read all images and store as numpy.array
	for filePath in filePaths:
		if ".png" in filePath:
			image = Image.open(
				path.join(filePath))
			images.append(np.array(image))

	images = np.array(images)

	return images

if __name__ == "__main__":
	print(readFolder("./"))