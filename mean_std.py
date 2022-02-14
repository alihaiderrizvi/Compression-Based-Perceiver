import os
from os.path import isfile, join
import numpy as np
import cv2

def list_subdir(dir):
	l = [x[0] for x in os.walk(dir)]
	return l[1:]

ans = list_subdir('custom_dataset_split/train')
lst = []
instances = []

for d in ans:
	onlyfiles = [f for f in os.listdir(d) if isfile(join(d, f))]
	for filepath in os.listdir(d):
		instances.append(cv2.imread(d + '/' + filepath))
	print(d)


instances = np.array(instances)
s = instances.shape
print(s)
# instances = instances / 255
# print('normalized', instances.shape)
instances = instances.reshape((s[0],s[1]*s[2],s[3]))
print(instances.shape)

mean1 = np.mean(instances, axis=1)
print(mean1.shape)
mean2 = np.mean(mean1, axis=0)
print(mean2.shape)
print(mean2)
'''
mean2 = np.array([104.66268819, 116.54580544, 118.23132099])
'''
'''
std = np.array([[0. for _ in range(3)] for _ in range(s[1]*s[2])])
c = 0
for img in instances:
	diff = np.square(img - mean2)
	std += diff
	c += 1
	print(std.shape, c)

std = np.mean(std, axis=0)
print(std)

std = std / (42000.*255.)
std = np.sqrt(std)
print(std.shape)
print(std)
'''
'''
std = np.array([4.37454361, 4.06389989, 4.11659655])
mean3 = mean2 / 255.
print(mean3)
print(std)
'''

