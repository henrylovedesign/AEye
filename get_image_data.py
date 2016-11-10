import os
import cv2

def getImageData(image_roots,gt_roots,trainfile,testfile):
	images = os.listdir(image_root)
	num=len(images)
	print num
	train=images[0:int(0.7*num)]
	test=images[int(0.7*num)::]
	print len(train)
	with open(trainfile,"w") as df: 
		for i in xrange(len(image_roots)):
			images = os.listdir(image_roots[i])
			num=len(images)
			print num
			train=images[0:int(0.7*num)]
			for image in train:
				image_path="/".join([image_root+image])
				print image_path
				rawimg = cv2.imread(image_path)
				try:
					w,h,c = rawimg.shape
					a = min(w,h)
					img = rawimg[0:a,0:a]
					img = cv2.resize(img,(40,40))

					cv2.imwrite(image_path,img)
					cv2.resize(img,(40,40))

					gt_path= "/".join([gt_roots[i]  , image+".txt"])
					try:
						label= open(gt_path,"r").read()[0].strip().split(" ")[0]

						df.write(" ".join([image_path, label  ]      )    )
						if image!=images[-1]:
							df.write("\n")
					except IOError:
						pass
				except AttributeError:
					pass

	with open(testfile,"w") as df: 
		for i in xrange(len(image_roots)):
			images = os.listdir(image_roots[i])
			num=len(images)
			print num
			test=images[int(0.7*num)::]

			for image in test:
				image_path="/".join([image_root+image])

				rawimg = cv2.imread(image_path)
				
				try:
					w,h,c = rawimg.shape
					a = min(w,h)
					img = rawimg[0:a,0:a]
					img = cv2.resize(img,(40,40))

					cv2.imwrite(image_path,img)
					cv2.resize(img,(40,40))

					gt_path= "/".join([gt_roots[i]  , image+".txt"])
					try:
						label= open(gt_path,"r").read()[0].strip().split(" ")[0]

						df.write(" ".join([image_path, label  ]      )    )
						if image!=images[-1]:
							df.write("\n")
					except IOError:
						pass
				except AttributeError:
					pass
	df.close()


image_roots=["/opt/henry/lablesnapshot/R2D2/train/","/opt/henry/lablesnapshot/r2d2_2/train"]
gt_roots=["/opt/henry/lablesnapshot/R2D2/R2D2GT","/opt/henry/lablesnapshot/r2d2_2/r2d2GT"]
trainfile="/home/henry/projects/toBeLable/R2D2/train.data"
testfile="/home/henry/projects/toBeLable/R2D2/test.data"
getImageData(image_roots,gt_roots,trainfile,testfile)
