import cv2
import os
import set_caffe_path
import caffe
import numpy as np
lenet_definition="/home/henry/projects/toBeLable/R2D2/r2d2_lenet/refine_deploy.prototxt"
weights="/home/henry/projects/toBeLable/R2D2/r2d2_lenet/refine_r2d2_lenet.caffemodel"
index=[]

for dog in os.listdir("/home/henry/projects/toBeLable/R2D2/dog/google"):
	dogim = "/home/henry/projects/toBeLable/R2D2/dog/google/"+dog
	test_image = cv2.imread(dogim)
	w,h,c=test_image.shape
	a = min(w,h)
	print a
	test_image = test_image[0:a,0:a] 
	print test_image.shape



	test_image = cv2.resize(test_image,(40,40))

	r2d2_lenet=caffe.Net(lenet_definition,weights,caffe.TEST)


	transformer = caffe.io.Transformer({'data': r2d2_lenet.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformed_image = transformer.preprocess('data', test_image)

	print transformed_image.shape
	r2d2_lenet.blobs['data'].data[...]=transformed_image
	print r2d2_lenet.blobs['data'].data.shape

	output = r2d2_lenet.forward()
	
	index.append(np.argmax(output['score']))

print index
print 1-float(sum(index))/len(index)