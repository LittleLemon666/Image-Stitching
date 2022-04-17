from math import sin, cos, atan2
from PIL import Image
import numpy as np
import os
from os import path
import argparse
from scipy.ndimage import gaussian_filter, sobel

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
	for x in range(output.shape[0]):
		for y in range(output.shape[1]):
			output[x][y] = image[x * s][y * s]
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

# for debugging
def showHarrisDetectorFeatures(image, p, threshold = 2.55, s = 2):
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			if (p[x // s, y // s] > threshold): # features
				# print(f"{y}, {x}")
				image[x, y, ...] = [0,0,255]
	image = Image.fromarray(image.astype(np.uint8))
	image.show()

# fill a color in r half-size rectangle, s is coordinate scale
def fillAreaValue(source, x, y, s, r, value):
	minx = max(x * s - r, 0)
	miny = max(y * s - r, 0)
	maxx = min(x * s + r, source.shape[0])
	maxy = min(y * s + r, source.shape[1])
	source[minx:maxx, miny:maxy] = value

def ANMS(p, r, n, threshold = 2.55):
	features = []
	while (len(features) < n and r > 1):
		r = r - 1
		p_r = np.copy(p)
		for f in features:
			fillAreaValue(p_r, f[0], f[1], 1, r, 0)
		max_value = threshold + 1
		while (max_value > threshold):
			xy = np.unravel_index(np.argmax(p_r, axis=None), p_r.shape)
			max_value = p_r[xy]
			features.append([xy[0], xy[1], max_value])
			fillAreaValue(p_r, xy[0], xy[1], 1, r, 0)
	return features

# for debugging
def showFeatures(image, features, s = 2):
	for feature in features:
		fillAreaValue(image, feature[0], feature[1], s, 1, [255, 0, 0])
	image = Image.fromarray(image.astype(np.uint8))
	image.show()

def getTranslateMatrix(x, y):
	m = np.matrix([[1, 0, x],
		 		   [0, 1, y],
		 		   [0, 0, 1]])
	return m

def getRotateMatrix(theta):
	m = np.matrix([[cos(theta), -sin(theta), 0],
		 		   [sin(theta), cos(theta), 0],
		 		   [0, 0, 1]])
	return m

def inverseWarping(source, affine):
	output = np.zeros(source.shape)
	affine_inverse = np.linalg.inv(affine)
	print(affine)
	for x in range(source.shape[0]):
		for y in range(source.shape[1]):
			coord = np.dot(affine_inverse, np.array([x, y, 1]))
			coord = [int(coord[0, 0]), int(coord[0, 1])]
			if coord[0] < 0 or coord[1] < 0 or coord[0] >= source.shape[0] or coord[1] >= source.shape[1]:
				continue
			output[coord[0], coord[1]] = source[x, y]
	return output

def testAffine(source):
	affine = np.dot(getTranslateMatrix(source.shape[0] // 2, source.shape[1] // 2),
						np.dot(getRotateMatrix(atan2(40,30)),
							   getTranslateMatrix(-source.shape[0] // 2, -source.shape[1] // 2)))
	image = inverseWarping(source, affine)
	image = Image.fromarray(image.astype(np.uint8))
	image.show()

def descript(source, pls, featuress):
	for level in range(len(featuress)):
		gx = sobel(pls[level], 0)
		gy = sobel(pls[level], 1)
		affine = np.dot(getTranslateMatrix(featuress[level][0][0], featuress[level][0][1]),
						np.dot(getRotateMatrix(atan2(gx[featuress[level][0][0], featuress[level][0][1]], gy[featuress[level][0][0], featuress[level][0][1]])),
							   getTranslateMatrix(-featuress[level][0][0], -featuress[level][0][1])))
		image = inverseWarping(source, affine)
		# image = Image.fromarray(image.astype(np.uint8))
		# image.show()
		# for feature in featuress[level]:
			# affineM = np.dot(getTranslateMatrix(-feature[0], -feature[1]), np.dot(getRotateMatrix(atan2(gx[feature[0]], gy[feature[1]])), getTranslateMatrix(feature[0], feature[1])))
	testAffine(image)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataPath", type=str,
						help="The directory of images", default="")
	args = parser.parse_args()
	images = readFolder(args.dataPath)
	r = 24
	feature_num = 500
	for i in range(1, 2): #len(images)
		I = toGrey(images[i])
		pls = []
		featuress = []
		hl = getHarrisDetector(I)
		pl = getPlprime(hl)
		features = ANMS(pl, r, feature_num)
		# showFeatures(images[i], features)
		pls.append(pl)
		featuress.append(features)
		
		for level in range(1, 4):
			hl = getHarrisDetector(pls[level - 1])
			pl = getPlprime(hl)
			# showHarrisDetectorFeatures(images[i], p0)
			features = ANMS(pl, r, feature_num)
			# showFeatures(images[i], features, (level + 1) * 2)
			pls.append(pl)
			featuress.append(features)

		descript(images[i], pls, featuress)
		# image = Image.fromarray(images[i].astype(np.uint8))
		# image.show()