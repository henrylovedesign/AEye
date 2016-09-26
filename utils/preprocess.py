import Image
import os
import selectivesearch
import sys
sys.path.insert(0,'/home/henry/softWare/caffe-master/python')

import caffe
project_root = '/home/henry/projects/artistEye/'

image_root = project_root +'image/'

def preprocess(image_root):
	
	imgListFile = project_root+'data/image.txt'
	roisListFile = project_root+'data/rois.txt'
	
	if os.path.exists(roisListFile):
		os.remove(roisListFile)

	if os.path.exists(imgListFile):
		os.remove(imgListFile)
	
	
	imageList = os.listdir(image_root)
	
	for image_name in imageList:
		
		img = Image.open(image_root+image_name)
		new_img = img.resize((256,192),Image.ANTIALIAS)
		image_path = image_root+image_name
		new_img.save(image_path)

		img = caffe.io.load_image(image_path)
		rois = [roi['rect'] for roi in selectivesearch.selective_search(img,scale=500, sigma=0.9, min_size=10)[1][:16]]		
		
		
		with open(project_root+'data/rois.txt','a') as roisFile:
			for roi in rois:
				xmin = str(roi[0])
				ymin = str(roi[1])
				xmax = str(roi[0]+roi[2])
				ymax = str(roi[0]+roi[3])
				roiInfor = ",".join([image_path,xmin,ymin,xmax,ymax])
				roisFile.write(roiInfor)
				if image_name != imageList[-1]:
					roisFile.write("\n")
				elif roi != rois[-1]:
					roisFile.write("\n")

			roisFile.close()
		
		with open(project_root+'data/image.txt','a') as imageFile:
                	imageFile.write(image_path)
			if image_name!=imageList[-1] :
				imageFile.write("\n")

			imageFile.close()
preprocess(image_root)
