	#flatten the large image and save it as npy file  
# use different image and caption with name and flatten it and store in numoy array and generate the training data
import cv2
import numpy as np
#initialize camera
cap=cv2.VideoCapture(0)
skip=0
face_data=[]
dataset_path="./data2/"
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
file_name=input("Enter the name of the person : ")
while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#gray image reduces complexity
	
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
	#area=width*height
	faces=sorted(faces,key=lambda f:f[2]*f[3])#you can use reverse=True or faces[-1:] both are same last image will be highest if reverse =true not written
	#print(faces)#returns tuple(x,y,w,h)
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)#thickness of rectangle
		offset=10#10 pixels adding on 4 sides of face
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))#resize in 100 *100 image
		skip+=1
		if (skip%10==0):
			face_data.append(face_section)
			print(len(face_data))
			
	cv2.imshow("Frame",frame)
	cv2.imshow("face_Section",face_section)

	key_pressed=cv2.waitKey(1) & 0xFF#wait for 1 ms and we use q key to stop#cv2.wait key rturns 32 bit or 64 bit according to the
	#platform and OXFF is 8 bit so result is 8 bit and compare to ord(q)
	if key_pressed==ord('q'):#ASCII value between 0 to 255#8 bit integer
		break
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))#no of columns to be figured out automatically
print(face_data.shape)#25,30000
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')
cap.release()#release the device
cv2.destroyAllWindows()
