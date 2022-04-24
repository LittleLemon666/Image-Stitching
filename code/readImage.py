from math import sin, cos, atan2
from PIL import Image, ImageDraw
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
		if ".jpg" in filePath.lower():
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
@njit
def fillAreaValue(source, y, x, s, r, value):
	miny = max(y * s - r, 0)
	minx = max(x * s - r, 0)
	maxy = min(y * s + r, source.shape[0])
	maxx = min(x * s + r, source.shape[1])
	source[miny:maxy, minx:maxx] = value

@njit
def my_unravel_index(v, shape):
	out = [0] * len(shape)
	for i in range(1, len(shape) + 1):
		out[-i] = v%shape[-i]
		v = v // shape[-i]
	return out

@njit
def ANMS(p, r, n, threshold = 3):
	features = []
	while (len(features) < n and r > 1):
		r = r - 1
		p_r = np.copy(p)
		for f in features:
			fillAreaValue(p_r, f[0], f[1], 1, r, 0)
		max_value = threshold + 1
		while (max_value > threshold):
			yx = my_unravel_index(np.argmax(p_r), p_r.shape)
			max_value = p_r[yx[0]][yx[1]]
			# for feature in features:
			#     if xy[0] == feature[0] and xy[1] == feature[1]:
			#         print([xy[0], xy[1], max_value])
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
	m = np.array([[1, 0, x],
					[0, 1, y],
					[0, 0, 1]], np.float)
	return m

def getRotateMatrix(theta):
	m = np.array([[cos(theta), -sin(theta), 0],
					[sin(theta), cos(theta), 0],
					[0, 0, 1]], np.float)
	return m

def getAffine(center_x, center_y, theta):
	return np.matmul(getTranslateMatrix(center_x, center_y),
					np.matmul(getRotateMatrix(theta),
						   getTranslateMatrix(-center_x, -center_y)))

def getArea(source, y, x, theta, r):
	l = 2 * r + 1
	area = np.zeros((l, l), np.float)

	affine = np.matmul(getTranslateMatrix(r - x, r - y), getAffine(x, y, theta))
	affine_inverse = np.linalg.inv(affine)

	return _getArea(source, affine_inverse, l, area)

@njit
def mySum(array):
	s = 0
	for a in range(len(array)):
		s += array[a]
	return s

@njit
def dotMatPos(mat, pos):
	return np.array([mySum([mat[v][x] * pos[x] if x < len(pos) else mat[v][x] for x in range(len(mat[v]))]) for v in range(len(pos))], np.float)

@njit
def _getArea(source, affine_inverse, l, area):
	out = False
	for y in range(l):
		for x in range(l):
			coord = dotMatPos(affine_inverse, np.array([x, y]))
			quad_base = np.floor(coord).astype(np.int32)
			ratio = 1 - (coord - quad_base)

			if quad_base[0] < 0 or quad_base[1] < 0 or quad_base[0] >= source.shape[1] - 1 or coord[1] >= source.shape[0] - 1:
				out = True
				continue
			area[y][x] =    source[quad_base[1],         quad_base[0]]         * ratio[0]             * ratio[1] +\
							source[quad_base[1] + 1,     quad_base[0]]         * ratio[0]             * (1 - ratio[1]) +\
							source[quad_base[1],         quad_base[0] + 1]     * (1 - ratio[0])    * ratio[1] +\
							source[quad_base[1] + 1,     quad_base[0] + 1]     * (1 - ratio[0])     * (1 - ratio[1])

	return area, out

def inverseWarping(canvas, source, affine):
	affine_inverse = np.linalg.inv(affine)
	boundary = np.array([
		[0, 0, 1], 
		[source.shape[1], 0, 1], 
		[0, source.shape[0], 1], 
		[source.shape[1], source.shape[0], 1]])
	newBoundary = np.matmul(affine, np.transpose(boundary))
	minX = math.floor(np.min(newBoundary[0]))
	maxX = math.ceil(np.max(newBoundary[0]))
	minY = math.floor(np.min(newBoundary[1]))
	maxY = math.ceil(np.max(newBoundary[1]))
	# print(minX, maxX, minY, maxY)
	_inverseWarping(canvas, source, affine_inverse, minX, maxX, minY, maxY)
	return canvas

@njit
def _inverseWarping(canvas, source, affine_inverse, minX, maxX, minY, maxY):
	for y in range(minY, maxY):
		for x in range(minX, maxX):
			coord = dotMatPos(affine_inverse, np.array([x, y]))
			quad_base = np.floor(coord).astype(np.int32)
			ratio = 1 - (coord - quad_base)

			if quad_base[0] < 0 or quad_base[1] < 0 or quad_base[0] >= source.shape[1] - 1 or coord[1] >= source.shape[0] - 1:
				continue
			canvas[y, x] = source[quad_base[1],         quad_base[0]]         * ratio[0]             * ratio[1] +\
							source[quad_base[1] + 1,     quad_base[0]]         * ratio[0]             * (1 - ratio[1]) +\
							source[quad_base[1],         quad_base[0] + 1]     * (1 - ratio[0])    * ratio[1] +\
							source[quad_base[1] + 1,     quad_base[0] + 1]     * (1 - ratio[0])     * (1 - ratio[1])

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
		polygon = [p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]]
		p0 = np.dot(affine, np.array([descriptor_x, descriptor_y - s * r, 1]))
		ImageDraw.Draw(image).polygon(polygon, outline="red")
		ImageDraw.Draw(image).line([descriptor_x, descriptor_y, p0[0], p0[1]], fill="red", width=1)
	image.show()


# descriptor: x y value gx gy normalisation
def descript(source, Is, pls, featuress):
	descriptors_level = []
	for level in range(len(featuress) - 1):
		descriptors = []
		gx = sobel(pls[level + 1], 0)
		gy = sobel(pls[level + 1], 1)
		for feature in featuress[level + 1]:
			center_y = int(feature[0])
			center_x = int(feature[1])
			theta = atan2(gy[center_y, center_x], gx[center_y, center_x])
			patch, out = getArea(Is[level + 1], center_y, center_x, theta, 20)
			if out:
				continue
			image = Image.fromarray(patch.astype(np.uint8))
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
			#print(f"x y theta: {descriptor[0]} {descriptor[1]} {theta}")
			# break
		descriptors_level.append(descriptors)
		# markDescriptors(source, descriptors, pow(2, level + 1))
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
	with open(args.dataPath + "/descriptors.json", "w") as f:
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

def showPair(image_a, image_b, pairs_level, descriptors_a, descriptors_b):
	pic = np.zeros((max(image_a.shape[0], image_b.shape[0]), image_a.shape[1] + image_b.shape[1], 3))
	pic[:image_a.shape[0], :image_a.shape[1]] = image_a
	pic[:image_b.shape[0], image_a.shape[1]:] = image_b
	image = Image.fromarray(pic.astype(np.uint8))
	for level in range(len(pairs_level)):
		s = pow(2, level + 1)
		pairs = pairs_level[level]
		for pair in pairs:
			point_a = descriptors_a[level][pair[0]]
			point_b = descriptors_b[level][pair[1]]
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
			points.append(np.array(descriptor[:2]) * pow(2, level + 1))
	return np.array(points)

def getNewCorners(shapeA, shapeB, transform):
	corners = np.zeros((8, 2), dtype=np.float)
	corners[0] = np.array([0, 0])
	corners[1] = np.array([shapeA[1], 0])
	corners[2] = np.array([0, shapeA[0]])
	corners[3] = np.array([shapeA[1], shapeA[0]])
	# print(transform[0])
	# print(transform[1])
	corners[4] = np.array([0, 0])
	corners[5] = np.array([shapeB[1], 0])
	corners[6] = np.array([0, shapeB[0]])
	corners[7] = np.array([shapeB[1], shapeB[0]])
	temp0 = np.matmul(corners[4:8], transform[0][:2]) + transform[0][2]
	temp1 = np.matmul(corners[4:8], transform[1][:2]) + transform[1][2]
	corners[4:8, 0] = temp0
	corners[4:8, 1] = temp1

	minX = math.floor(np.min(corners[:, 0]))
	maxX = math.ceil(np.max(corners[:, 0]))
	minY = math.floor(np.min(corners[:, 1]))
	maxY = math.ceil(np.max(corners[:, 1]))
	return minX, maxX, minY, maxY

def ransac(descriptorsA, descriptorsB, pairs, k, m, outlierDistance):
	newPairs = np.zeros((sum([len(pairs) for pairs in pairs]), 2), np.int64)
	offset = 0
	for level in range(len(pairs)):
		_pairs = pairs[level]
		_pairs = np.array(_pairs)
		_pairs[:, 0] += sum([len(descriptorsA[a]) for a in range(0, level)])
		_pairs[:, 1] += sum([len(descriptorsB[a]) for a in range(0, level)])
		if len(_pairs) > 0:
			newPairs[offset:offset+len(_pairs)] = _pairs
			offset += len(_pairs)

	pointsA = getPointsInDescriptors(descriptorsLevels=descriptorsA)[newPairs[:, 0]]
	pointsB = getPointsInDescriptors(descriptorsLevels=descriptorsB)[newPairs[:, 1]]
	temp = np.zeros((len(pointsA), 4), np.float)
	temp[:, :2] = pointsA
	temp[:, 2:] = pointsB

	inlierCounts = []
	transforms = []
	for i in range(k):
		chosenPairs = np.random.choice(newPairs.shape[0], m, replace=False)
		transform = findTransform(pointsA[chosenPairs], pointsB[chosenPairs])

		newPointsB = np.zeros(pointsB.shape)
		newPointsB[:, 0] = np.matmul(pointsB, transform[:2]) + transform[2]
		newPointsB[:, 1] = np.matmul(pointsB, transform[3:5]) + transform[5]

		delta = newPointsB - pointsA
		delta2 = delta * delta
		distance = np.sum(delta2, axis=1)

		inlierCount = np.count_nonzero(distance < outlierDistance * outlierDistance)
		inlierCounts.append(inlierCount)

		mat = np.array([
			transform[:3],
			transform[3:],
			[0, 0, 1]
		], np.float)
		transforms.append(mat)
	# print(inlierCounts)
	index = np.argmax(inlierCounts)
	if np.max(inlierCounts) == 0:
		print("Wrong outlierDistance!!!!")
	return transforms[index]


def getChainedTransform(matches, transforms, a):
	if matches[a] == -1:
		return np.array([
			[1, 0, 0],
			[0, 1, 0],
			[0, 0 ,1]
		], dtype=np.float)
	else:
		return np.matmul(getChainedTransform(matches, transforms, matches[a]), transforms[a])

def alignImages(images, descriptors, matches, k, m, outlierDistance, threshold = 0.65):

	transforms = []

	for i in range(len(matches)):
		if matches[i] == -1:
			transforms.append(0)
			continue
		descriptorsA = descriptors[matches[i]]
		descriptorsB = descriptors[i]

		pairs = featureMatch(descriptorsA, descriptorsB, threshold)

		# showPair(images[matches[i]], images[i], pairs, descriptorsA, descriptorsB)

		transform = ransac(descriptorsA, descriptorsB, pairs, k, m, outlierDistance)
		transforms.append(transform)

	print(matches)

	outImages = [0] * len(images)

	finished = [False] * len(images)
	offset = [[0, 0]] * len(images)
	changed = True
	while changed:
		changed = False
		for i in range(len(matches)):
			if not finished[i]:
				if matches[i] == -1:
					outImages[i] = images[i]
					finished[i] = True
					changed = True
				elif finished[matches[i]]:
					root = findRoot(matches, i)

					transform = getChainedTransform(matches, transforms, i)
					transform[0][2] += offset[root][0]
					transform[1][2] += offset[root][1]
					print(transform)

					minX, maxX, minY, maxY = getNewCorners(outImages[root].shape, images[i].shape, transform)
					newImage = np.zeros((maxY - minY, maxX - minX, 3), dtype=np.float32)
					newImage[-minY:-minY + outImages[root].shape[0], -minX:-minX + outImages[root].shape[1]] = outImages[root]
					mat = np.array([
						[transform[0][0], transform[0][1], transform[0][2] - minX],
						[transform[1][0], transform[1][1], transform[1][2] - minY],
						[0, 0, 1]
					], np.float)
					newImage = inverseWarping(newImage, images[i], mat)

					offset[root][0] += -minX
					offset[root][1] += -minY
					outImages[root] = newImage
					finished[i] = True
					changed = True

	return outImages

def getPairCount(pairs_level):
	return sum([len(pairs) for pairs in pairs_level])

def findRoot(matches, a):
	for i in range(len(matches)):
		if matches[a] == -1:
			return a
		else: 
			a = matches[a]
	return -1

def checkSameRoot(matches, a, b):
	root_a = findRoot(matches, a)
	root_b = findRoot(matches, b)
	if root_a == -1 or root_b == -1:
		return False
	return root_a == root_b

def getTree(matches, a):
	inTree = [False] * len(matches)
	root = findRoot(matches, a)

	for i in range(len(matches)):
		if inTree[i]:
			continue
		if findRoot(matches, matches[i]) == root:
			inTree[i] = True
	
	out = []
	for i in range(len(inTree)):
		if inTree[i]:
			out.append(i)

	return out

def setRoot(matches, a):
	originalRoot = findRoot(matches, a)
	matches[a] = -1

	targets = getTree(matches, originalRoot)
	connections = []
	for i in range(len(targets)):
		connections.append([targets, matches[targets]])
	
	target = [a]
	changed = True
	while changed:
		changed = False
		for i in range(len(connections)):
			if connections[i][0] in target:
				matches[connections[i][1]] = connections[i][0]
				target.append(connections[i][1])
				changed = True
			elif connections[i][1] in target:
				matches[connections[i][0]] = connections[i][1]
				target.append(connections[i][0])
				changed = True
	return matches


def getMatch(descriptors, match_threshold = 0.65, threshold = 30):
	l = len(descriptors)
	match_mat = np.zeros((l, l), dtype=np.uint8)
	for i in range(l - 1):
		for j in range(i + 1, l - 1):
			pairs_level = featureMatch(descriptors[j], descriptors[i], match_threshold)
			match_mat[i][j] = getPairCount(pairs_level)
	
	matches = [-1] * l
	max_value = 0

	while True:
		yx = my_unravel_index(np.argmax(match_mat), match_mat.shape)
		max_value = match_mat[yx[0]][yx[1]]
		if max_value == 0 or max_value < threshold:
			break
		
		print(max_value)
		if checkSameRoot(matches, yx[0], yx[1]):
			pass
		elif matches[yx[0]] == -1:
			matches[yx[0]] = yx[1]
		elif matches[yx[1]] == -1:
			matches[yx[1]] = yx[0]
		else:
			a = len(getTree(matches, yx[0]))
			b = len(getTree(matches, yx[1]))
			if a > b:
				matches = setRoot(matches, yx[1])
				matches[yx[1]] = yx[0]
			else:
				matches = setRoot(matches, yx[0])
				matches[yx[0]] = yx[1]
		match_mat[yx[0]][yx[1]] = 0
	return matches

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataPath", type=str,
						help="The directory of images", default="")
	parser.add_argument("-j", "--descriptorsPath", action='store_true',
						help="The path of descriptors")
	args = parser.parse_args()
	images = readFolder(args.dataPath)
	r = 24
	feature_num = 1000
	image_descriptors = []
	level_num = 3
	if args.descriptorsPath:
		image_descriptors = readDescriptors(args.dataPath + "/descriptors.json")
		# for i in range(1, 3): #len(images)
		#     for level in range(0, level_num - 1):
		#         markDescriptors(images[i], image_descriptors[i - 1][level], pow(2, level + 1))

		feature_match_threshold = 0.65
		match_threshold = 20
		matches = getMatch(image_descriptors, feature_match_threshold, match_threshold)

		subarasiiImages = alignImages(images, image_descriptors, matches, 1000, match_threshold, 7, feature_match_threshold)
		for i in range(len(subarasiiImages)):
			if type(subarasiiImages[i]) is np.ndarray:
				Image.fromarray(subarasiiImages[i].astype(np.uint8)).save(f"temp{i}.png")


	else:
		for i in range(len(images)):
			I = toGrey(images[i])
			Is = [I]
			pls = []
			featuress = []
			p0 = getHarrisDetector(I)
			features = ANMS(p0, r, feature_num)
			pls.append(p0)
			featuress.append(features)
			# showFeatures(images[i], features, 1)
			
			for level in range(1, level_num):
				I = getPlprime(Is[level - 1])
				pl = getHarrisDetector(I)
				# showHarrisDetectorFeatures(images[i], p0)
				features = ANMS(pl, r, feature_num)
				Is.append(I)
				pls.append(pl)
				featuress.append(features)
				# showFeatures(images[i], features, pow(2, level))
				# showFeatures(I, features, 1)

			# testAffine(images[i])
			image_descriptor = descript(images[i], Is, pls, featuress)
			image_descriptors.append(image_descriptor)
			print(f"{i + 1} / {len(images)}")
		
		saveDescriptors(image_descriptors)