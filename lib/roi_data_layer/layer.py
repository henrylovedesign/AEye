from random import shuffle
import sys
caffe_root = "/home/henry/softWare/caffe-master/python"
sys.path.insert(0,caffe_root)
import caffe
import numpy as np
import yaml
import csv

class RoIDataLayer(caffe.Layer):

	def setup(self,bottom,top):
		
		self.top_names_map = {'data':0,'rois':1}
		
		layer_params = yaml.load(self.param_str)		
		
		self.batch_size = layer_params['batch_size']
		self.rois_number = layer_params['rois_number']
		
		top[0].reshape(self.batch_size,3,256,192)
		top[1].reshape(self.batch_size*self.rois_number,5)

		self.imagesource = layer_params['imagesource']
		self.roisource = layer_params['roisource']

	def forward(self,bottom,top):
		"""
        	Load data.
        	"""
		 # Use the batch loader to load the next image.
		im,rois= load_next_batches(self.imagesource,self.roisource,self.batch_size)
        	

		for itt in range(self.batch_size):
            	
            	# Add directly to the caffe data layer
            		top[0].data[itt, ...] = im[itt]
			for roiitt in range(self.rois_number):
            			top[1].data[sel.rois_number*itt+roiitt, ...] = rois[self.rois_number*itt+roiitt]
	def backward(self,bottom,top):
		pass

	def reshape(self,bottom,top):
		pass
	
def load_next_batches(imagesSourceFile,roiSource,number,roisNumber):
	
	images = [line.rstrip() for line in open(imagesSourceFile)]
	
	extractImages = shuffle(images)[:number]
	
	rois = [line.rstrip() for line in open(roiSourc)]
	
	extractImagesRoi=[]
	for itt in range(number):
		for roi in rois:
			infor = roi.split(",")	
			if(infor[0].strip() == extractImages[itt]):
				xmin = float(infor[1])
				ymin = float(infor[2])
				xmax = float(infor[3])
				ymax = float(infor[4])
			
				extractImagesRoi.append([itt,xmin,ymin,xmax,ymax])
	extractImagesRoi = np.asarrary(extractImagesRoi)
	ims = np.zeros((number,192,256,3))
	for itt in range(number):
		ims[itt] = caffe.io.load_image(extractImages[0])

	return ims,extractImagesRoi
		

