from PIL import Image
import numpy as np
import os
from os import path
import argparse
from scipy.ndimage import gaussian_filter

from numba import njit

@njit
def toGrey(image):
	return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114

def getHarrisDetector(I, sigma = 1.5):
	Ix = np.zeros(I.shape)
	gaussian_filter(I, (sigma,sigma), (0, 1), Ix)
	Iy = np.zeros(I.shape)
	gaussian_filter(I, (sigma,sigma), (1, 0), Iy)
	Sx2 = gaussian_filter(Ix**2, sigma)
	Sxy = gaussian_filter(Ix*Iy, sigma)
	Sy2 = gaussian_filter(Iy**2, sigma)
	detM = Sx2 * Sy2 - Sxy**2
	trM = Sx2 + Sy2 + 1e-6
	output = detM / trM
	return output

def getPlprime(image, s = 2, sigma = 1.0):
	image = gaussian_filter(image, sigma)
	output = np.zeros((int(image.shape[0] / s), int(image.shape[1] / s)))
	for y in range(output.shape[0]):
		for x in range(output.shape[1]):
			output[y][x] = image[y * s][x * s]
	return output

def readFolder(folderPath):

	images = []
	filePaths = os.listdir(folderPath)

	# read all images and store as numpy.array
	for filePath in filePaths:
		if ".jpg" in filePath:
			image = Image.open(
				path.join(folderPath, filePath))
			images.append(np.array(image))

	images = np.array(images)

	return images

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataPath", type=str,
						help="The directory of images", default="")
	args = parser.parse_args()
	images = readFolder(args.dataPath)
	p0s = []
	for i in range(1,2): #len(images)
		I = toGrey(images[i])
		hl = getHarrisDetector(I)
		p0 = getPlprime(hl)
		for y in range(images[i].shape[0]):
			for x in range(images[i].shape[1]):
				if (p0[int(y / 2), int(x / 2)] > 2.55): # features
					# print(f"{y}, {x}")
					images[i][y,x,...] = [0,0,255]
		image = Image.fromarray(images[i].astype(np.uint8))
		image.show()
		p0s.append(p0)
	
	