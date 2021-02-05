import cv2
import numpy as np

X = []
Y = []

for i in range(1,462):

	img = cv2.imread('images/img_'+str(i)+'.jpg')
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Removing noise
	#img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	
	x_s = 512/img.shape[1]
	y_s = 512/img.shape[0]

	resized_img = cv2.resize(img,(512,512))

	file = "ground_truth/gt_img_"+str(i)+".txt"

	labels = np.zeros((16,16,5))
	
	with open(file,'r') as f:
		for row in f:
			line = row.split(',')
			x1 = int(int(line[0].strip())*x_s)
			y1 = int(int(line[1].strip())*y_s)
			x2 = int(int(line[2].strip())*x_s)
			y2 = int(int(line[3].strip())*y_s)
			center_x = (x1+x2)//2
			center_y = (y1+y2)//2
			w = abs(x2-x1)
			h = abs(y2-y1)
			l = np.array([1,center_x,center_y,w,h])
			labels[center_y//32][center_x//32] = l
			

			#cv2.rectangle(resized_img,(int(x*x1),int(y*y1)),(int(x*x2),int(y*y2)),(255,0,0))
			#cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))

	X.append(resized_img)
	Y.append(labels)

X = np.asarray(X)
print(X.shape)
Y = np.asarray(Y)
print(Y.shape)
	