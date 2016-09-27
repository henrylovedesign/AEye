import caffe
import numpy as np
import csv

class NWSwapLayer(caffe.Layer):

        def setup(self,bottom,top):
                pass
        def forward(self,bottom,top):
                
	

		top_temp = bottom[0].data
		top_temp = np.transpose(top_temp,(0,2,1,3))
		top[0].reshape(*(top_temp.shape))
		size = top_temp.shape[0]
		for itt in range(size):
			top[0].data[itt,...]= top_temp[itt]

        def backward(self,top,propagate_down,bottom):
		diff_temp = np.transpose(self.diff,(2,1,0,3))
		size = diff_temp.shape[0]
		for itt in range(size):
                	bottome[0].diff[itt,...] = diiff_temp[itt]
        def reshape(self,bottom,top):
                top[0].reshape(*(np.transpose(bottom[0].data,(2,1,0,3)).shape))
		self.diff =  np.zeros(np.transpose(bottom[0].data,(2,1,0,3)).shape)

