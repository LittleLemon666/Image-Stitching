from math import sin, cos, atan2
from PIL import Image, ImageDraw
from matplotlib.cbook import flatten
import numpy as np
import os
from os import path
import argparse
from scipy.ndimage import gaussian_filter, sobel
from sklearn.neighbors import NearestNeighbors
import json

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
def fillAreaValue(source, y, x, s, r, value):
	miny = max(y * s - r, 0)
	minx = max(x * s - r, 0)
	maxy = min(y * s + r, source.shape[0])
	maxx = min(x * s + r, source.shape[1])
	source[miny:maxy, minx:maxx] = value

def ANMS(p, r, n, threshold = 2.55):
	features = []
	while (len(features) < n and r > 1):
		r = r - 1
		p_r = np.copy(p)
		for f in features:
			fillAreaValue(p_r, f[0], f[1], 1, r, 0)
		max_value = threshold + 1
		while (max_value > threshold):
			yx = np.unravel_index(np.argmax(p_r, axis=None), p_r.shape)
			max_value = p_r[yx]
			# for feature in features:
			# 	if xy[0] == feature[0] and xy[1] == feature[1]:
			# 		print([xy[0], xy[1], max_value])
			features.append([yx[0], yx[1], max_value])
			fillAreaValue(p_r, yx[0], yx[1], 1, r, 0)
	return features

# for debugging
def showFeatures(image, features, s = 2):
	image = np.copy(image)
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

def getAffine(center_x, center_y, theta):
	return np.dot(getTranslateMatrix(center_x, center_y),
					np.dot(getRotateMatrix(theta),
						   getTranslateMatrix(-center_x, -center_y)))

def inverseWarping(source, affine):
	output = np.zeros(source.shape)
	affine_inverse = np.linalg.inv(affine)
	for y in range(source.shape[0]):
		for x in range(source.shape[1]):
			coord = np.dot(affine_inverse, np.array([x, y, 1]))
			coord = [int(coord[0, 0]), int(coord[0, 1])]
			if coord[0] < 0 or coord[1] < 0 or coord[0] >= source.shape[1] or coord[1] >= source.shape[0]:
				continue
			output[y, x] = source[coord[1], coord[0]]
	return output

def testAffine(source):
	affine = getAffine(source.shape[1] // 2, source.shape[0] // 2, atan2(40,30))
	image = inverseWarping(source, affine)
	image = Image.fromarray(image.astype(np.uint8))
	image.show()

def testMarkFeature(source):
	image = Image.fromarray(source)
	polygon = [5, 5, 100, 5, 100, 100, 5, 100]
	ImageDraw.Draw(image).polygon(polygon, outline="red")
	image.show()

# the direction is up if theta = 0.
# theta is between -PI/2 ~ PI/2
def markDescriptors(source, descriptors, s = 1, r = 20):
	image = Image.fromarray(source)
	for descriptor in descriptors:
		descriptor_x = s * descriptor[0] 
		descriptor_y = s * descriptor[1] 
		affine = getAffine(descriptor_x, descriptor_y, atan2(descriptor[4] , descriptor[3]))
		p1 = np.dot(affine, np.array([descriptor_x - s * r, descriptor_y - s * r, 1]))
		p2 = np.dot(affine, np.array([descriptor_x + s * r, descriptor_y - s * r, 1]))
		p3 = np.dot(affine, np.array([descriptor_x + s * r, descriptor_y + s * r, 1]))
		p4 = np.dot(affine, np.array([descriptor_x - s * r, descriptor_y + s * r, 1]))
		polygon = [p1[0,0], p1[0,1], p2[0,0], p2[0,1], p3[0,0], p3[0,1], p4[0,0], p4[0,1]]
		p0 = np.dot(affine, np.array([descriptor_x, descriptor_y - s * r, 1]))
		ImageDraw.Draw(image).polygon(polygon, outline="red")
		ImageDraw.Draw(image).line([descriptor_x, descriptor_y, p0[0,0], p0[0,1]], fill="red", width=1)
	image.show()

@njit
def getArea(source, y, x, s, r):
	miny = max(y * s - r, 0)
	minx = max(x * s - r, 0)
	maxy = min(y * s + r, source.shape[0])
	maxx = min(x * s + r, source.shape[1])
	return source[miny:maxy, minx:maxx]

# descriptor: x y value gx gy normalisation
def descript(source, Is, pls, featuress):
	descriptors_level = []
	for level in range(len(featuress)):
		descriptors = []
		gy = sobel(pls[level], 0)
		gx = sobel(pls[level], 1)
		for feature in featuress[level]:
			center_y = feature[0]
			center_x = feature[1]
			theta = atan2(gy[center_y, center_x], gx[center_y, center_x])
			affine = getAffine(center_x, center_y, theta)
			image = inverseWarping(Is[level], affine)
			patch = getArea(image, center_y, center_x, 1, 20)
			patch = gaussian_filter(patch, 4.5)
			patch = Image.fromarray(patch.astype(np.uint8))
			patch = patch.resize((8, 8))
			# patch.show()
			normalisation = (patch - np.mean(patch)) / np.std(patch)
			descriptor = [center_x, center_y, feature[2], gx[center_y, center_x], gy[center_y, center_x], normalisation]
			descriptors.append(descriptor)
			print(f"x y value: {descriptor[0]} {descriptor[1]} {descriptor[2]}")
		descriptors_level.append(descriptors)
		markDescriptors(source, descriptors, pow(2, level))
	# testMarkFeature(source)
	return descriptors_level

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def saveDescriptors(image_descriptors):
	with open("./descriptors.json", "w") as f:
		json.dump(image_descriptors, f, cls=NpEncoder)

def readDescriptors(path):
	image_descriptors = []
	with open(path, "r") as f:
		image_descriptors = json.loads(f.read())
	return image_descriptors

def flattenDescriptorsLevel(descriptors_level):
	flattenss_level = []
	for descriptors in descriptors_level:
		flattenss = []
		for descriptor in descriptors:
			flattens = [float(descriptor[0]), float(descriptor[1]), float(descriptor[2]), float(descriptor[3]), float(descriptor[4])]
			descriptor[5] = np.array(descriptor[5])
			for y in range(len(descriptor[5])):
				for x in range(len(descriptor[5][y])):
					flattens.append(float(descriptor[5][y][x]))
			flattenss.append(flattens)
		flattenss_level.append(flattenss)
		# flatten = [attribute for descriptor in descriptors for attribute in descriptor]
			# flattens.append(flatten)
	flattenss_level = np.array(flattenss_level, dtype=np.float64)
	return flattenss_level

def flattenDescriptors(descriptors):
	flattenss = []
	for descriptor in descriptors:
		flattens = [float(descriptor[0]), float(descriptor[1]), float(descriptor[2]), float(descriptor[3]), float(descriptor[4])]
		descriptor[5] = np.array(descriptor[5])
		for y in range(len(descriptor[5])):
			for x in range(len(descriptor[5][y])):
				flattens.append(float(descriptor[5][y][x]))
		flattenss.append(flattens)
		# flatten = [attribute for descriptor in descriptors for attribute in descriptor]
			# flattens.append(flatten)
	flattenss = np.array(flattenss, dtype=np.float64)
	return flattenss

def featureMatch(descriptorsA, descriptorsB):
	pairLevel = []
	for level in range(len(descriptorsA)):
		A = flattenDescriptors(descriptorsA[level])
		B = flattenDescriptors(descriptorsB[level])
		target = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(A)
		distances, indices = target.kneighbors(B)
		print(distances)
		print("-------")
		print(indices)
		pair = []
		
		pairLevel.append(pair)

	return pairLevel

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataPath", type=str,
						help="The directory of images", default="")
	parser.add_argument("-j", "--descriptorsPath", type=str,
						help="The path of descriptors", default="")
	args = parser.parse_args()
	images = readFolder(args.dataPath)
	r = 24
	feature_num = 250
	image_descriptors = []
	if args.descriptorsPath:
		image_descriptors = readDescriptors("./descriptors.json")
		for i in range(1, 3): #len(images)
			for level in range(0, 2):
				markDescriptors(images[i], image_descriptors[i - 1][level], pow(2, level))

		# flattenDescriptors(image_descriptors[0])
		# for descriptors_level in image_descriptors:
			# descriptors_level = flattenDescriptors(descriptors_level)
		# print(image_descriptors[0])

		for i in range(1, 2): #len(images) - 1
			featureMatch(image_descriptors[i - 1], image_descriptors[i])
	
	else:
		for i in range(1, 3): #len(images)
			I = toGrey(images[i])
			Is = [I]
			pls = []
			featuress = []
			p0 = getHarrisDetector(I)
			features = ANMS(p0, r, feature_num)
			pls.append(p0)
			featuress.append(features)
			
			for level in range(1, 2):
				I = getPlprime(Is[level - 1])
				pl = getHarrisDetector(I)
				# showHarrisDetectorFeatures(images[i], p0)
				features = ANMS(pl, r, feature_num)
				Is.append(I)
				pls.append(pl)
				featuress.append(features)

			# testAffine(images[i])
			image_descriptor = descript(images[i], Is, pls, featuress)
			image_descriptors.append(image_descriptor)
		
		saveDescriptors(image_descriptors)