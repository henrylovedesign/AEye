from random import shuffle
import sys
caffe_root = "/home/henry/projects/artistEye/caffe-fast-rcnn/py-fast-rcnn/caffe-fast-rcnn"
sys.path.insert(0,caffe_root)
import caffe
import numpy as np
import yaml
import csv

class RoIDataLayer(caffe.Layer):

	def setup(self,bottom,top):
		
		self.top_names_map = {'data':0,'rois':1}
		
		layer_params = yaml.load(self.param_str_)		
		print(layer_params)	
		self.batch_size = layer_params['batch_size']
		self.rois_number = layer_params['rois_number']
		
		top[0].reshape(self.batch_size,3,192,256)
		top[1].reshape(self.batch_size*self.rois_number,5)

		self.imagesource = layer_params['imagesource']
		self.roisource = layer_params['roisource']

	def forward(self,bottom,top):
		"""
        	Load data.
        	"""
		 # Use the batch loader to load the next image.
		im,rois= load_next_batches(self.imagesource,self.roisource,self.batch_size,self.rois_number)
        	

		for itt in range(self.batch_size):
            	
            	# Add directly to the caffe data layer
            		top[0].data[itt, ...] = im[itt]
			for roiitt in range(self.rois_number):
            			top[1].data[self.rois_number*itt+roiitt, ...] = rois[self.rois_number*itt+roiitt]
	def backward(self,bottom,top):
		pass

	def reshape(self,bottom,top):
		pass
	
def load_next_batches(imagesSourceFile,roiSource,number,roisNumber):
	
	images = [line.rstrip() for line in open(imagesSourceFile)]
	
	shuffle(images)
	
	extractImages = images[:number]
	
	rois = [line.rstrip() for line in open(roiSource)]
	
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
	extractImagesRoi = np.asarray(extractImagesRoi)
	ims = np.zeros((number,3,192,256))
	for itt in range(number):
		ims[itt] = np.transpose(caffe.io.load_image(extractImages[0])*255,(2,0,1))

	return ims,extractImagesRoi
		

