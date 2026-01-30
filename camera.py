import cv2, threading, time, numpy as np, subprocess, os

class CameraStream:
    def __init__(self, src=0, resolution=(640, 480), fps=24):
        self.src, self.resolution, self.fps = src, resolution, fps
        self.stopped, self.frame, self.grabbed, self.use_fallback = False, None, False, False
        self._init_camera()

    def _init_camera(self):
        # Primary: libcamera/rpicam pipe (High Performance for Pi)
        self.bin_path = "/usr/bin/rpicam-vid" if os.path.exists("/usr/bin/rpicam-vid") else "/usr/bin/libcamera-vid"
        
        if os.path.exists(self.bin_path):
            print(f"📸 Starting MJPEG Pipe: {self.bin_path}", flush=True)
            cmd = [self.bin_path, "-t", "0", "--width", str(self.resolution[0]), 
                   "--height", str(self.resolution[1]), "--framerate", str(self.fps), 
                   "--codec", "mjpeg", "--nopreview", "-o", "-"]
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)
                time.sleep(1)
                if self.proc.poll() is None: return
            except: pass
            
        self._init_fallback()

    def _init_fallback(self):
        print(f"💡 Switching to OpenCV VideoCapture({self.src})...", flush=True)
        self.use_fallback = True
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.use_fallback:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret: self.frame, self.grabbed = frame, True
                time.sleep(1/self.fps)
            else:
                self._update_pipe()

    def _update_pipe(self):
        stream_bytes = b""
        while not self.stopped and not self.use_fallback:
            try:
                chunk = self.proc.stdout.read(16384)
                if not chunk: break
                stream_bytes += chunk
                while True:
                    a = stream_bytes.find(b'\xff\xd8')
                    b = stream_bytes.find(b'\xff\xd9', a)
                    if a != -1 and b != -1:
                        jpg = stream_bytes[a:b+2]
                        stream_bytes = stream_bytes[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None: self.frame, self.grabbed = frame, True
                    else: break
            except: break

    def read(self):
        if self.frame is not None: return self.frame.copy()
        return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

    def stop(self):
        self.stopped = True
        if not self.use_fallback: self.proc.terminate()
        else: self.cap.release()
