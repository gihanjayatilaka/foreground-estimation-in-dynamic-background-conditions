'''
INTERFACE TO BE USED FOR CO227 project in 2018
@Harshana please implement this
'''


def readFrame(fileName):
	'''
	String fileName is the file name of the JPEG
	Should return a uint8[H][W][3] array of the frame
	'''


def appendFrame(fileName,fr):
	'''
	String fileName.avi is the output file name
	fr = uint[H][W] array which represents a greyscale image
	
	If fileName does not exist the avi video file should be created with one frame.
	If the fileName exists, the fr frame should be appended to the video file

	'''


def newFigure3D():
	'''
	This function should clear the existsing figure or create a new figure
	This figure will be used for plotting things in 3D RGB space
	'''

def plot(x,y,z):
	'''
	x,y,z= uint
	This should plot a point on (x,y,z)
		
	'''


def plotSphere(centerX,centerY,centerZ,radius):
	'''
	This should plot a sphere
	'''


def plotCylinder(centerX,centerY,centerZ,radius,halfLength):
	'''
	This should plot a cylinder
	'''
