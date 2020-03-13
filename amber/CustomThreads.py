import time
import picamera
import picamera.array
from threading import Thread

""" A class to run camera I/O on a different thread """
class ThreadedCamera:
	""" Set up class data """
	def __init__(self):
	# Set up class data
		self.frame = None
		self.stopped = False
		self.camera = None
		self.stream = None
		pass
    
	""" Start the camera stream and thread it """
	def start(self):
		# Set up the camera and stream
		self.camera = picamera.PiCamera()
		self.camera.resolution = (2592, 1936)
		self.camera.start_preview()
		time.sleep(0.5)
        
		self.stream = picamera.array.PiRGBArray(self.camera)
		self.camera.capture(self.stream, format='bgr')
		self.frame = self.stream.array
                    
		# Start the thread
		Thread(target=self.update, args=()).start()
		return self
        
	""" Set the class's frame to the stream array """
	def update(self):
        
		while(True):
            
			# Loop exit case
			if self.stopped:
				return
                    
			# At this point the image is available as stream.array
			self.frame = self.stream.array
    
	""" Get the class's frame (from the stream) """
	def read(self):
		return self.frame
    
	""" Tell update to stop """
	def stop(self):
		self.stopped = True
		return
