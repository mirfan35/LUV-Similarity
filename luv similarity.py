import numpy as np 
import cv2

##################################################################################
# convert bgr to image
################################################################################## 
def colour2img(b,g,r,h=1,w=1):
	res = np.empty((h,w,3), dtype=np.uint8)
	res[:,:,0] = np.full((h,w), b, dtype=np.uint8)
	res[:,:,1] = np.full((h,w), g, dtype=np.uint8)
	res[:,:,2] = np.full((h,w), r, dtype=np.uint8)
	return res

##################################################################################
# Luv Similarity 
##################################################################################
Cd = [ 183.31934916,  -83.3193807,    30.50476572,   25.4389103,  -120.5271033]
Ca = [ 1.12522175, -0.47827603,  0.04969178,  0.04356436]
def LuvSimilarity(img1, img2):
	arr1 = cv2.cvtColor(np.float32(img1)/255, cv2.COLOR_BGR2Luv)
	arr2 = cv2.cvtColor(np.float32(img2)/255, cv2.COLOR_BGR2Luv)
	arr = (arr1+arr2)/2

	## euclidean similarity
	dis = np.sqrt(np.sum((arr1 - arr2)**2, axis=2))
	ndis = np.sqrt(np.sum((arr)**2, axis=2))+0.00000001
	l,u,v = arr[:,:,0]/ndis, arr[:,:,1]/ndis, arr[:,:,2]/ndis
	dref =  Cd[0]+Cd[1]*l+Cd[2]*u+Cd[3]*v+Cd[4]*l*u*v
	euc = 1-(dis/dref+0.00000001)
	euc[euc<0] = 0
	
	## chroma 
	chr1 = np.sqrt(np.sum(arr1[:,:,1:3]**2,axis=2))
	chr2 = np.sqrt(np.sum(arr2[:,:,1:3]**2,axis=2))
	ch = np.tanh(chr1/2.9)*np.tanh(chr2/2.9)

	## Hue
	Hue = (np.arctan2(arr1[:,:,2], arr1[:,:,1]) - np.arctan2(arr2[:,:,2], arr2[:,:,1]))
	tet = ch*np.abs((Hue + np.pi) % (2*np.pi) - np.pi)

	## angular similarity
	ang = np.arctan2(arr[:,:,2], arr[:,:,1])
	aref =  Ca[0]+Ca[1]*ang+Ca[2]*ang**2+Ca[3]*ang**3 

	tet = 1-(tet/aref)
	tet[tet<0] = 0
	return euc*tet

def sigmoid(x, a=1, b=0, c=1, d=0):
	return a/(1+np.exp((b-x)/c))+d
	
##################################################################################
# main
################################################################################## 
img1 = cv2.imread('colourStrip.png') # read image

b,g,r = 0,255,255 # BGR value

img2 = colour2img(b,g,r, h=img1.shape[0], w=img1.shape[1]) # convert b,g,r into image

res = LuvSimilarity(img1, img2) # camparing colour between two images

## display result
cv2.imshow('image 1',img1)
cv2.imshow('image 2',img2)
cv2.imshow('similarity result', res)
cv2.waitKey(0)
