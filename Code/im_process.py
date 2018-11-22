from sklearn.feature_extraction import image as IMG
import numpy as np
import cv2
from utils import integrate_images

#extract patches from the image for all scales of the image
#return the INTEGREATED images and the coordinates of the patches
def image2patches(scales, image, patch_w = 16, patch_h = 16):
	all_patches = np.zeros((0, patch_h, patch_w))
	all_x1y1x2y2 = []
	for s in scales:
		simage = cv2.resize(image, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
		height, width = simage.shape
		print('Image shape is: %d X %d' % (height, width))
		patches = IMG.extract_patches_2d(simage, (patch_w, patch_h)) # move along the row first

		total_patch = patches.shape[0]
		row_patch = (height - patch_h + 1)
		col_patch = (width - patch_w + 1)
		assert(total_patch == row_patch * col_patch)
		scale_xyxy = []
		for pid in range(total_patch):
			y1 = pid / col_patch
			x1 = pid % col_patch
			y2 = y1 + patch_h - 1
			x2 = x1 + patch_w - 1
			scale_xyxy.append([int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)])
		all_patches = np.concatenate((all_patches, patches), axis = 0)
		all_x1y1x2y2 += scale_xyxy
	return integrate_images(normalize(all_patches)), all_x1y1x2y2

#return a vector of prediction (0/1) after nms, same length as scores
#input: [x1, y1, x2, y2, score], threshold used for nms
#output: [x1, y1, x2, y2, score] after nms
def nms(xyxys, overlap_thresh):
	######################
	######## TODO ########
	######################
	x1,y1,x2,y2,scores=xyxys[:,0],xyxys[:,1],xyxys[:,2],xyxys[:,3],xyxys[:,4]
	area=(x2-x1+1)*(y2-y1+1)
	pick=[]
	idx=np.argsort(scores)
	while len(idx)>0:
		last=len(idx)-1
		i=idx[last]
		pick.append(i)
		suppress=[last]
		for pos in np.arange(0,last):
			j=idx[pos]
			xx1=max(x1[i],x1[j])
			yy1=max(y1[i],y1[j])
			xx2=min(x2[i],x2[j])
			yy2=min(y2[i],y2[j])
			w=max(0,xx2-xx1+1)
			h=max(0,yy2-yy1+1)
			overlap=float(w*h)/area[j]
			if overlap>overlap_thresh:
				suppress.append(pos)
		idx=np.delete(idx,suppress)
	pred=xyxys[pick,:]
	return pred

def normalize(images):
	standard = np.std(images)
	images = (images - np.min(images)) / (np.max(images) - np.min(images))
	return images

def main():
	original_img = cv2.imread('Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	scales = 1 / np.linspace(1, 10, 46)
	patches, patch_xyxy = image2patches(scales, original_img)
	print(patches.shape)
	print(len(patch_xyxy))
if __name__ == '__main__':
	main()
