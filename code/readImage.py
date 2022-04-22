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

import math
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

def ANMS(p, r, n, threshold = 3):
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
def showFeatures(image, features, s = 1):
	# print(image.shape)
	image = np.copy(image)
	if len(image.shape) == 2:
		output = np.zeros((image.shape[0], image.shape[1], 3))
		for c in range(3):
			output[:, :, c] = image[:, :]
		image = np.copy(output)
		# target = np.zeros(image.shape[0], image.shape[1], 3)
		# target[:,:,...] = [image[:,:], image[:,:], image[:,:]]
		# imagetarget
		# image = image[:, :, np.newaxis]
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

def inverseWarping2(canvas, source, affine):
	affine_inverse = np.linalg.inv(affine)
	for y in range(canvas.shape[0]):
		for x in range(canvas.shape[1]):
			coord = np.matmul(affine_inverse, np.array([x, y, 1]))
			coord = [int(coord[0]), int(coord[1])]
			if coord[0] < 0 or coord[1] < 0 or coord[0] >= source.shape[1] or coord[1] >= source.shape[0]:
				continue
			canvas[y, x] = source[coord[1], coord[0]]
	return canvas

def testAffine(source):
	affine = getAffine(source.shape[1] // 2, source.shape[0] // 2, atan2(40,10))
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
	for level in range(len(featuress) - 1):
		descriptors = []
		gx = sobel(pls[level + 1], 0)
		gy = sobel(pls[level + 1], 1)
		for feature in featuress[level + 1]:
			center_y = feature[0]
			center_x = feature[1]
			theta = atan2(gy[center_y, center_x], gx[center_y, center_x])
			affine = getAffine(center_x, center_y, theta)
			image = inverseWarping(Is[level + 1], affine)
			patch = getArea(image, center_y, center_x, 1, 20)
			# image = Image.fromarray(patch.astype(np.uint8))
			# image.show()
			patch = gaussian_filter(patch, 1) #4.5
			patch = Image.fromarray(patch.astype(np.uint8))
			#patch.show()
			patch = patch.resize((8, 8))
			#patch.show()
			patch = np.asarray(patch, dtype=np.float).T
			normalisation = (patch - np.mean(patch)) / np.std(patch)
			descriptor = [center_x, center_y, feature[2], gx[center_y, center_x], gy[center_y, center_x], normalisation]
			descriptors.append(descriptor)
			print(f"x y theta: {descriptor[0]} {descriptor[1]} {theta}")
			# break
		descriptors_level.append(descriptors)
		markDescriptors(source, descriptors, pow(2, level + 1))
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
			flattens = []
			descriptor[5] = np.array(descriptor[5])
			for y in range(len(descriptor[5])):
				for x in range(len(descriptor[5][y])):
					flattens.append(float(descriptor[5][y][x]))
			flattenss.append(flattens)
		flattenss_level.append(flattenss)
	flattenss_level = np.array(flattenss_level, dtype=np.float64)
	return flattenss_level

def flattenDescriptors(descriptors):
	flattenss = []
	for descriptor in descriptors:
		flattens = []
		descriptor[5] = np.array(descriptor[5])
		for y in range(len(descriptor[5])):
			for x in range(len(descriptor[5][y])):
				flattens.append(float(descriptor[5][y][x]))
		flattenss.append(flattens)
	flattenss = np.array(flattenss, dtype=np.float64)
	return flattenss

def featureMatch(descriptorsA, descriptorsB, threshold = 0.6):
	pairs_level = []
	for level in range(len(descriptorsA)):
		A = flattenDescriptors(descriptorsA[level])
		B = flattenDescriptors(descriptorsB[level])
		target = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(B)
		distances, indices = target.kneighbors(A)
		# print(distances)
		# print("-------")
		# print(indices)
		pairs = []
		for i in range(len(indices)):
			if distances[i][0] < distances[i][1] * threshold:
				pairs.append([i, indices[i][0]])
		pairs_level.append(pairs)

	return pairs_level

def showPair(image_a, image_b, pairs, descriptors_a, descriptors_b, s):
	pic = np.zeros((max(image_a.shape[0], image_b.shape[0]), image_a.shape[1] + image_b.shape[1], 3))
	pic[:image_a.shape[0], :image_a.shape[1]] = image_a
	pic[:image_b.shape[0], image_a.shape[1]:] = image_b
	image = Image.fromarray(pic.astype(np.uint8))
	for pair in pairs:
		print(pair)
		point_a = descriptors_a[pair[0]]
		point_b = descriptors_b[pair[1]]
		ImageDraw.Draw(image).ellipse([point_a[0] * s - 5, point_a[1] * s - 5, point_a[0] * s + 5, point_a[1] * s + 5], fill ="red", outline ="red")
		ImageDraw.Draw(image).ellipse([image_a.shape[1] + point_b[0] * s - 5, point_b[1] * s - 5, image_a.shape[1] + point_b[0] * s + 5, point_b[1] * s + 5], fill ="red", outline ="red")
		ImageDraw.Draw(image).line([point_a[0] * s, point_a[1] * s, image_a.shape[1] + point_b[0] * s, point_b[1] * s], fill="blue", width=1)
	image.show()

def findTransform(pointsA, pointsB):
	pointsLen = min(pointsA.shape[0], pointsB.shape[0])
	A = np.zeros((pointsLen * 2, 6), dtype=np.float)
	b = np.zeros((pointsLen * 2), dtype=np.float)

	for i in range(pointsLen):
		A[i * 2][0] = pointsB[i][0]
		A[i * 2][1] = pointsB[i][1]
		A[i * 2][2] = 1
		b[i * 2] = pointsA[i][0]

		A[i * 2 + 1][3] = pointsB[i][0]
		A[i * 2 + 1][4] = pointsB[i][1]
		A[i * 2 + 1][5] = 1
		b[i * 2 + 1] = pointsA[i][1]

	x = np.linalg.lstsq(A,b, rcond=None)
	# x_B * x1 + y_B * x2 + x3 = x_A
	# x_B * x4 + y_B * x5 + x6 = y_A
	return x[0]

def getPointsInDescriptors(descriptorsLevels):
	points = []
	for level in range(len(descriptorsLevels)):
		for descriptor in descriptorsLevels[level]:
			points.append(np.array(descriptor[:2]) * pow(2, level))
	return np.array(points)

def ransac(descriptorsA, descriptorsB, pairs, k, m, outlierDistance):
	pointsA = getPointsInDescriptors(descriptorsLevels=descriptorsA)[pairs[:, 0]]
	pointsB = getPointsInDescriptors(descriptorsLevels=descriptorsB)[pairs[:, 1]]

	print(pointsA)
	print(pointsB)

	inlierCounts = []
	transforms = []
	for i in range(k):
		chosenPairs = np.random.choice(pairs.shape[0], m, replace=False)
		transform = findTransform(pointsA[chosenPairs], pointsB[chosenPairs])

		newPointsB = np.zeros(pointsB.shape)
		newPointsB[:, 0] = np.matmul(pointsB, transform[:2]) + transform[2]
		newPointsB[:, 1] = np.matmul(pointsB, transform[3:5]) + transform[5]

		delta = newPointsB - pointsA
		delta2 = delta * delta
		distance = np.sum(delta2, axis=1)

		inlierCount = np.count_nonzero(distance < outlierDistance * outlierDistance)
		inlierCounts.append(inlierCount)
		transforms.append(transform)
	print(inlierCounts)
	index = np.argmax(inlierCounts)
	if np.max(inlierCounts) == 0:
		print("Wrong outlierDistance!!!!")
	return transforms[index]

def alignImages(imageA, imageB, descriptorsA, descriptorsB, pairs, k, m, outlierDistance):
	corners = np.zeros((8, 2), dtype=np.float)
	corners[0] = np.array([0, 0])
	corners[1] = np.array([imageA.shape[1], 0])
	corners[2] = np.array([0, imageA.shape[0]])
	corners[3] = np.array([imageA.shape[1], imageA.shape[0]])

	transform = ransac(descriptorsA, descriptorsB, pairs, k, m, outlierDistance)
	print(transform[:3])
	print(transform[3:])
	corners[4] = np.array([0, 0])
	corners[5] = np.array([imageB.shape[1], 0])
	corners[6] = np.array([0, imageB.shape[0]])
	corners[7] = np.array([imageB.shape[1], imageB.shape[0]])
	temp0 = np.matmul(corners[4:8], transform[:2]) + transform[2]
	temp1 = np.matmul(corners[4:8], transform[3:5]) + transform[5]
	corners[4:8, 0] = temp0
	corners[4:8, 1] = temp1

	minX = math.floor(np.min(corners[:, 0]))
	maxX = math.ceil(np.max(corners[:, 0]))
	minY = math.floor(np.min(corners[:, 1]))
	maxY = math.ceil(np.max(corners[:, 1]))

	newImage = np.zeros((maxY - minY, maxX - minX, 3), dtype=np.float)
	newImage[-minY:-minY + imageA.shape[0], -minX:-minX + imageA.shape[1]] = imageA
	mat = np.array([
		[transform[0], transform[1], transform[2] - minX],
		[transform[3], transform[4], transform[5] - minY],
		[0, 0, 1]
	], np.float)
	inverseWarping2(newImage, imageB, mat)
	return newImage

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
	level_num = 3
	if args.descriptorsPath:
		image_descriptors = readDescriptors("./descriptors.json")
		for i in range(1, 3): #len(images)
			for level in range(0, level_num - 1):
				markDescriptors(images[i], image_descriptors[i - 1][level], pow(2, level + 1))

		for i in range(1, 2): #len(images) - 1
			pairs_level = featureMatch(image_descriptors[i - 1], image_descriptors[i], 0.6)
			for level in range(0, level_num - 1):
				showPair(images[i], images[i + 1], pairs_level[level], image_descriptors[i - 1][level], image_descriptors[i][level], pow(2, level + 1))

			newPairs = np.zeros((sum([len(pairs) for pairs in pairs_level]), 2), np.int64)
			offset = 0
			for _pairs in pairs_level:
				pairs = np.array(_pairs)
				if len(pairs) > 0:
					newPairs[offset:offset+len(pairs)] = pairs + offset
					offset += len(pairs)

			newImage = alignImages(images[i], images[i+1], image_descriptors[i - 1], image_descriptors[i], newPairs, 100, 10, 30)
			Image.fromarray(newImage.astype(np.uint8)).save("temp.png")


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
			showFeatures(images[i], features, 1)
			
			for level in range(1, level_num):
				I = getPlprime(Is[level - 1])
				pl = getHarrisDetector(I)
				# showHarrisDetectorFeatures(images[i], p0)
				features = ANMS(pl, r, feature_num)
				Is.append(I)
				pls.append(pl)
				featuress.append(features)
				showFeatures(images[i], features, pow(2, level))
				# showFeatures(I, features, 1)

			# testAffine(images[i])
			image_descriptor = descript(images[i], Is, pls, featuress)
			image_descriptors.append(image_descriptor)
		
		saveDescriptors(image_descriptors)