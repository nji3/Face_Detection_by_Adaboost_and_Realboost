from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed

class Weak_Classifier(ABC):
	#initialize a harr filter with the positive and negative rects
	#rects are in the form of [x1, y1, x2, y2] 0-index
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		self.id = id
		self.plus_rects = plus_rects
		self.minus_rects = minus_rects
		self.num_bins = num_bins
		self.activations = None

	#take in one integrated image and return the value after applying the image
	#integrated_image is a 2D np array
	#return value is the number BEFORE polarity is applied
	def apply_filter2image(self, integrated_image):
		pos = 0
		for rect in self.plus_rects:
			rect = [int(n) for n in rect]
			pos += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		neg = 0
		for rect in self.minus_rects:
			rect = [int(n) for n in rect]
			neg += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		return pos - neg
	
		
	#take in a list of integrated images and calculate values for each image
	#integrated images are passed in as a 3-D np-array
	#calculate activations for all images BEFORE polarity is applied
	#only need to be called once
	def apply_filter(self, integrated_images):
		values = []
		for idx in range(integrated_images.shape[0]):
			values.append(self.apply_filter2image(integrated_images[idx, ...]))
		if (self.id + 1) % 100 == 0:
			print('Weak Classifier No. %d has finished applying' % (self.id + 1))
		return values
	
	#using this function to compute the error of
	#applying this weak classifier to the dataset given current weights
	#return the error and potentially other identifier of this weak classifier
	#detailed implementation is up you and depends
	#your implementation of Boosting_Classifier.train()
	@abstractmethod
	def calc_error(self, weights, labels):
		pass
	
	@abstractmethod
	def predict_image(self, integrated_image):
		pass

class Ada_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.polarity = None
		self.threshold = None
		self.emp_error = 1

	def calc_error(self, weights, labels):
		######################
		######## TODO ########
		######################
		activations = np.array(self.activations)
		min_bond,max_bond = min(activations),max(activations)
		rand_threshold = np.linspace(min_bond,max_bond,self.num_bins)
		label = np.asarray(labels)
		self.emp_error = 1
		for thred in rand_threshold:
			polarity = 1
			emp_label = polarity*np.sign(activations-thred)
			err = np.sum(weights[emp_label!=label])

			if err>0.5: 
				polarity = -1
				emp_label = polarity*np.sign(activations-thred)
				err = np.sum(weights[emp_label!=label])
			
			if err < self.emp_error:
				self.emp_error = err
				self.threshold = thred
				self.polarity = polarity
		return self.emp_error
		
	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		return self.polarity * np.sign(value - self.threshold)

class Real_Weak_Classifier(Weak_Classifier):
	
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.thresholds = None #this is different from threshold in ada_weak_classifier, think about it
		self.bin_pqs = None
		self.train_assignment = None

	def calc_error(self, weights, labels):
		######################
		######## TODO ########
		######################
		activations = np.sort(self.activations)
		sort_index = np.argsort(self.activations)
		weights = weights[sort_index]
		labels = labels[sort_index]
		indexs = np.array([i for i in range(len(labels))])
		bins = np.array_split(indexs,self.num_bins)
		weighted_labels = weights*labels
		bin_ps = np.zeros(self.num_bins)
		bin_qs = np.zeros(self.num_bins)
		for i in range(self.num_bins):
			bin_ps[i] = np.sum(weights[bins[i]]*(weighted_labels[bins[i]]>0))+1e-5
			bin_qs[i] = np.sum(weights[bins[i]]*(weighted_labels[bins[i]]<0))+1e-5
		self.bin_pqs = np.vstack((bin_ps,bin_qs))
		bins_threds = np.array([max(b) for b in bins])
		self.thresholds = activations[bins_threds]
		self.train_assignment = [np.sum(self.thresholds<activation) for activation in self.activations]

		return self.train_assignment,self.bin_pqs
	
	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		bin_idx = np.sum(self.thresholds < value)
		return 0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])

def main():
	plus_rects = [(1, 2, 3, 4)]
	minus_rects = [(4, 5, 6, 7)]
	num_bins = 50
#	ada_hf = Ada_Weak_Classifier(plus_rects, minus_rects, num_bins)
#	real_hf = Real_Weak_Classifier(plus_rects, minus_rects, num_bins)

if __name__ == '__main__':
	main()
