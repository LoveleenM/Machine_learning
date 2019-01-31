import cv2
import numpy as np
import os
def distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())
def knn(train,test,k=5):
	dist=[]
	for i in range(train.shape[0]):
		ix=train[i,:-1]
		iy=train[i,-1]
		d=distance(test,ix)
		dist.append([d,iy])
	dk=sorted(dist,key=lambda x:x[0])[:k]
	labels=np.array(dk)[:,-1]
	output=np.unique(labels,return_counts=True)#return  the label and their occurences
	index=np.argmax(output[1])#return the max occurences index
	return output[0][index]

cap=cv2.VideoCapture(0)
skip=0
face_data=[]#x vaues of data
label=[]#y values of data2
class_id=0#labels for files#the files are loaded
names={}#mapping between id and name
dataset_path="./data2/"
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

for fx in os.listdir(dataset_path):#all files present in directory that is data2 folder
	if fx.endswith('.npy'):
		names[class_id]=fx[:-4]
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)

		target=class_id*np.ones((data_item.shape[0],))#data_item.shape[0] means the no of pictures of that particular person so we are assifgning 
		#class id to the all photographs to that person
		class_id+=1
		label.append(target)
face_dataset=np.concatenate(face_data,axis=0)#in a single error
print(face_dataset.shape)
label_dataset=np.concatenate(label,axis=0).reshape((-1,1))
print(label_dataset.shape)
trainset=np.concatenate((face_dataset,label_dataset),axis=1)
print(trainset.shape)



while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	for face in faces[-1:]:
		x,y,w,h=face
		offset=10#10 pixels adding on 4 sides of face
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))#resize in 100 *100 image
		out=knn(trainset,face_section.flatten())#you can use reshape
		pred_name=names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)#thickness of rectangle
	cv2.imshow("faces",frame)
	key_pressed=cv2.waitKey(1) & 0xFF#wait for 1 ms and we use q key to stop#cv2.wait key rturns 32 bit or 64 bit according to the
	#platform and OXFF is 8 bit so result is 8 bit and compare to ord(q)
	if key_pressed==ord('q'):#ASCII value between 0 to 255#8 bit integer
		break
cap.release()#release the device
cv2.destroyAllWindows()





