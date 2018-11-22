import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None

		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
 
	
	#select weak classifiers to form a strong classifier
	#after training, by calling self.sc_function(), a prediction can be made
	#self.chosen_wcs should be assigned a value after self.train() finishes
	#call Weak_Classifier.calc_error() in this function
	#cache training results to self.visualizer for visualization
	#
	#
	#detailed implementation is up to you
	#consider caching partial results and using parallel computing
	def train(self, save_dir = None, load_dir = None):
		######################
		######## TODO ########
		######################

		if self.style == 'Ada':
			if load_dir is not None and os.path.exists(load_dir):
				print('[Find cached chosen classifiers, %s loading...]' % load_dir)
				with open(load_dir, 'rb') as f:
					load_wcs = pickle.load(f)
				self.chosen_wcs = load_wcs
			else:
				epochs = self.num_chosen_wc
				wcs_errors = []
				chosen_wcs = []

				weights = np.ones(len(self.data))/len(self.data)
				error = np.zeros(epochs)
				alpha = np.zeros(epochs)
				h = np.zeros(len(self.data))

				for t in range(epochs):
					errors = Parallel(n_jobs = self.num_cores)(delayed(wc.calc_error)(weights,self.labels) for wc in self.weak_classifiers)
					wcs_errors.append(errors)
					wcs = self.weak_classifiers[errors.index(min(errors))]
					wcs.id = np.argmin(np.array(errors))
					wcs.calc_error(weights,self.labels)
					error[t] = min(errors)
					alpha[t] = 1/2*np.log((1-error[t])/error[t])
					chosen_wcs.append([alpha[t],wcs])
					#sc = np.array([self.sc_function(image) for image in self.data])
					activ = np.asarray(wcs.activations)
					h[activ>=wcs.threshold],h[activ<wcs.threshold]=wcs.polarity,-wcs.polarity
					weights = weights*np.exp(-h*self.labels*alpha[t])
					weights = weights/sum(weights)
			
					print('In epochs %d, Training Error is %f' %(t,error[t]))
				self.chosen_wcs = chosen_wcs

				if save_dir is not None:
					pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))
				np.save('wcs_errors_45k.npy',wcs_errors)
		
		elif self.style == 'Real':
			if load_dir is not None and os.path.exists(load_dir):
				print('[Find cached chosen classifiers, %s loading...]' % load_dir)
				with open(load_dir, 'rb') as f:
					load_wcs = pickle.load(f)
			chosen_ids = [wc.id for alpha,wc in load_wcs]
			chosen_wcs = [self.weak_classifiers[id] for id in chosen_ids]
			weights = np.ones(len(self.data))/len(self.data)
			h_ts = []
			for t in range(self.num_chosen_wc):
				assignment_t,bin_pqs_t = chosen_wcs[t].calc_error(weights,self.labels)
				Z_t = 2*np.sum(np.sqrt(bin_pqs_t[0,:]*bin_pqs_t[1,:]))
				h_b = 0.5*np.log(bin_pqs_t[0,:]/bin_pqs_t[1,:])
				h_t = np.array([h_b[assign] for assign in assignment_t])
				weights = 1/Z_t*weights*np.exp(-self.labels*h_t)
				h_ts.append(h_t)
				pred_labels = np.sign(np.sum(np.array(h_ts),axis=0))
				err = np.sum(pred_labels!=self.labels)/len(self.labels)
				print('In the epoch %d, the Strong classifers Error is %f' %(t,err))
			np.save('Real_boost_hts.npy',h_ts)
			self.chosen_wcs = chosen_wcs

	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, name, scale_step = 20):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		#train_predicts = []
		#for idx in range(self.data.shape[0]):
		#	train_predicts.append(self.sc_function(self.data[idx, ...]))
		#print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################
		print(scale_step)
		scales = 1 / np.linspace(1, 8, scale_step)
		if os.path.exists('patches_%s.npy' %name):
			patches = np.load('patches_%s.npy' %name)
			patch_xyxy = np.load('patch_position_%s.npy' %name)
			print('Patches loaded')
		else:
			patches, patch_xyxy = image2patches(scales, img)
			print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
			np.save('patches_%s.npy' %name, patches)
			np.save('patch_position_%s.npy' %name, patch_xyxy)
			print('Patches saved')

		if os.path.exists('patches_score%s.pkl' %name):
			print('[Find cached Patches Scores, patches_score%s.pkl loading...]' % name)
			with open('patches_score%s.pkl' %name, 'rb') as f:
				predicts = pickle.load(f)
			print('Patches Scores loaded')
		else:
			predicts = [self.sc_function(patch) for patch in tqdm(patches)]
			pickle.dump(predicts, open('patches_score%s.pkl' %name, 'wb'))
			print('Patches Scores saved')

		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([np.hstack((patch_xyxy[idx], np.array(score))) for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms:', xyxy_after_nms.shape[0],xyxy_after_nms.shape[1])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3

		return img

	def get_hard_negative_patches(self, img, name, scale_step = 10):

		scales = 1 / np.linspace(1, 8, scale_step)
		if os.path.exists('patches_%s.npy' %name):
			patches = np.load('patches_%s.npy' %name)
			patch_xyxy = np.load('patch_position_%s.npy' %name)
			print('Patches loaded')
		else:
			patches, patch_xyxy = image2patches(scales, img)
			print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
			np.save('patches_%s.npy' %name, patches)
			np.save('patch_position_%s.npy' %name, patch_xyxy)
			print('Patches saved')

		if os.path.exists('patches_score%s.pkl' %name):
			print('[Find cached Patches Scores, patches_score%s.pkl loading...]' % name)
			with open('patches_score%s.pkl' %name, 'rb') as f:
				predicts = pickle.load(f)
			print('Patches Scores loaded')
		else:
			predicts = [self.sc_function(patch) for patch in tqdm(patches)]
			pickle.dump(predicts, open('patches_score%s.pkl' %name, 'wb'))
			print('Patches Scores saved')

		predicts = np.array(predicts)
		wrong_patches = patches[np.where(predicts > 0), ...]

		return wrong_patches

	def visualize(self):
		self.visualizer.labels = self.labels
		
		## Plot the 1000 weak classifiers
		T1 = [0,9,49,99]
		wcs_errors = np.load('wcs_errors.npy')
		wcs_err_reduce = []
		for t in T1:
			wcs_errors[t].sort()
			wcs_err_reduce.append(wcs_errors[t][0:999])
		self.visualizer.weak_classifier_accuracies = wcs_err_reduce
		self.visualizer.draw_wc_accuracies()

		## Plot the histograms
		T2 = [9,49,99]
		score = []
		chosen_wcs = self.chosen_wcs
		for t in T2:
			self.chosen_wcs = chosen_wcs[0:t]
			score.append(np.array([self.sc_function(d) for d in self.data]))
		self.visualizer.strong_classifier_scores = score
		self.visualizer.draw_histograms()

		## Plot the ROC
		self.visualizer.draw_rocs()

		## Plot the Top 20 Haar Filters
		self.visualizer.filters = self.chosen_wcs
		self.visualizer.top_filters()
