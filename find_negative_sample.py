import os
import cv2

def getImageData(image_root,gt_root,file):
	images = os.listdir(image_root)
	with open(file,"w") as df: 
		for image in images:
			image_path="/".join([image_root+image])
			rawimg = cv2.imread(image_path)
			try:
				w,h,c = rawimg.shape
				a = min(w,h)
				img = rawimg[0:a,0:a]
				img = cv2.resize(img,(40,40))

				cv2.imwrite(image_path,img)
				cv2.resize(img,(40,40))

				gt_path= "/".join([gt_root  , image+".txt"])
				try:
					label= open(gt_path,"r").read()[0].strip().split(" ")[0]
					print label
					if int(label)==0:
						df.write(" ".join([image_path, label  ]      )    )
						if image!=images[-1]:
							df.write("\n")
				except IOError:
					pass
			except AttributeError:
				pass


image_root="/home/henry/projects/toBeLable/R2D2/R2D2Image/"
gt_root="/home/henry/projects/toBeLable/R2D2/R2D2GT/"
file="/home/henry/projects/toBeLable/R2D2/negative.data"
getImageData(image_root,gt_root,file)