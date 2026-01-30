import cv2
import threading
import time
import numpy as np

class CameraStream:
    def __init__(self, src=0, resolution=(640, 480), fps=30):
        self.src = src
        self.resolution = resolution
        self.fps = fps
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.stream = None
        
        # Initialize camera
        self._init_camera()

    def _init_camera(self):
        print(f"📹 [SYS] Initializing imaging array at source {self.src}...")
        # On Pi, CAP_V4L2 is usually best
        backends = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY]
        for backend in backends:
            self.stream = cv2.VideoCapture(self.src, backend)
            if self.stream.isOpened():
                print(f"✅ [SYS] Sensor hooked via backend {backend}")
                break
        
        if self.stream.isOpened():
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.stream.set(cv2.CAP_PROP_FPS, self.fps)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0) # Warmup
            
            # Read first frame
            self.grabbed, self.frame = self.stream.read()
            if self.grabbed:
                print("✅ Camera started successfully")
            else:
                print("⚠️ Camera opened but failed to read frame")
        else:
            print("❌ Failed to open camera")

    def start(self):
        """Start the thread to read frames from the video stream."""
        if not self.stream or not self.stream.isOpened():
            print("⚠️ Camera not ready, cannot start thread.")
            return self
            
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Keep looping infinitely until the thread is stopped."""
        while True:
            if self.stopped:
                self.stream.release()
                return

            if self.stream.isOpened():
                grabbed, frame = self.stream.read()
                if grabbed:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    # If read fails, wait a bit and try again
                    time.sleep(0.1)
            else:
                time.sleep(1.0) # Wait before retry if stream closed

    def read(self):
        """Return the most recent frame."""
        if self.frame is not None:
             return self.frame.copy() # Return copy to prevent race conditions
        
        # Return a mock frame if no frame is available
        return self._get_mock_frame()

    def _get_mock_frame(self):
        img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        cv2.putText(img, "No Camera Signal", (int(self.resolution[0]/4), int(self.resolution[1]/2)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    def stop(self):
        """Indicate that the thread should be stopped."""
        self.stopped = True
