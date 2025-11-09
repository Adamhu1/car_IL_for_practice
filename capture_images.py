import cv2
import threading
import time

class SynchronizedDualCamera:
    def __init__(self, cam1_id=0, cam2_id=1):
        self.cam1 = cv2.VideoCapture(cam1_id)
        self.cam2 = cv2.VideoCapture(cam2_id)
        self.frame1 = None
        self.frame2 = None
        self.lock = threading.Lock()
        
    def start_capture(self):
        """启动相机捕获线程"""
        self.running = True
        self.thread1 = threading.Thread(target=self._capture_cam1)
        self.thread2 = threading.Thread(target=self._capture_cam2)
        self.thread1.start()
        self.thread2.start()
    
    def _capture_cam1(self):
        while self.running:
            ret, frame = self.cam1.read()
            if ret:
                with self.lock:
                    self.frame1 = frame
    
    def _capture_cam2(self):
        while self.running:
            ret, frame = self.cam2.read()
            if ret:
                with self.lock:
                    self.frame2 = frame
    
    def get_frames(self):
        """获取同步的帧"""
        with self.lock:
            return self.frame1, self.frame2
    
    def stop_capture(self):
        """停止捕获"""
        self.running = False
        self.thread1.join()
        self.thread2.join()
        self.cam1.release()
        self.cam2.release()

# 使用示例
def main():
    cam = SynchronizedDualCamera(0, 1)
    cam.start_capture()
    
    try:
        while True:
            frame1, frame2 = cam.get_frames()
            if frame1 is not None and frame2 is not None:
                cv2.imshow('Left', frame1)
                cv2.imshow('Right', frame2)
                
                key = cv2.waitKey(1)
                if key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"sync_left_{timestamp}.jpg", frame1)
                    cv2.imwrite(f"sync_right_{timestamp}.jpg", frame2)
                    print("同步拍摄完成！")
                elif key == ord('q'):
                    break
    finally:
        cam.stop_capture()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()