import cv2
import numpy as np
import time
from collections import defaultdict

class RobotRaceTracker:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video device")
            
        # Get initial frame dimensions
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.aspect_ratio = self.original_width / self.original_height
        
        # Finish line parameters (relative to original dimensions)
        self.finish_line_y = int(self.original_height * 0.5)
        self.finish_line_x1 = int(self.original_width * 0.2)
        self.finish_line_x2 = int(self.original_width * 0.8)
        self.finish_line_thickness = 2
        self.finish_line_angle = 0
        
        # Color ranges and tracking variables (same as before)
        self.color_ranges = {
            'red': ([0, 150, 100], [10, 255, 255]),
            'blue': ([100, 150, 100], [130, 255, 255]),
            # ... (keep your existing color definitions)
        }
        
        self.robot_positions = {}
        self.finish_times = {}
        self.has_crossed = defaultdict(bool)
        self.start_time = time.time()
        self.race_started = False
        self.leaderboard = []
        self.paused = True
        self.setup_mode = True
        self.line_confirmed = False
        
        # Create resizable window
        cv2.namedWindow('Robot Race Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Robot Race Tracker', self.original_width, self.original_height)
        
    def resize_frame(self, frame, target_width, target_height):
        """Resize frame while maintaining aspect ratio"""
        h, w = frame.shape[:2]
        
        # Calculate new dimensions maintaining aspect ratio
        if w/h > target_width/target_height:
            new_w = target_width
            new_h = int(target_width / self.aspect_ratio)
        else:
            new_h = target_height
            new_w = int(target_height * self.aspect_ratio)
            
        # Resize the frame
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create black background and center the resized frame
        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return background, x_offset, y_offset, new_w, new_h
    
    def get_finish_line_points(self):
        """Calculate finish line endpoints in original coordinates"""
        # ... (keep your existing implementation)
    
    def detect_robots(self, frame):
        """Detect robots in original frame coordinates"""
        # ... (keep your existing implementation)
    
    def check_finish_line_crossing(self, positions):
        """Check crossings in original coordinates"""
        # ... (keep your existing implementation)
    
    def display_leaderboard(self, frame):
        """Display leaderboard on the displayed frame"""
        # ... (keep your existing implementation)
    
    def display_instructions(self, frame):
        """Display instructions on the displayed frame"""
        # ... (keep your existing implementation)
    
    def reset_race(self):
        """Reset race state"""
        # ... (keep your existing implementation)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Process frame in original resolution for detection
            processed_frame = frame.copy()
            line_p1, line_p2 = self.get_finish_line_points()
            cv2.line(processed_frame, line_p1, line_p2, 
                     (0, 0, 255) if not self.line_confirmed else (0, 255, 0), 
                     self.finish_line_thickness)
            
            if not self.paused and not self.setup_mode:
                positions = self.detect_robots(processed_frame)
                if positions:
                    if not self.race_started:
                        self.race_started = True
                        print("Race started!")
                    self.check_finish_line_crossing(positions)
            
            # Get current window size
            win_w = cv2.getWindowImageRect('Robot Race Tracker')[2]
            win_h = cv2.getWindowImageRect('Robot Race Tracker')[3]
            
            # Resize the display frame to fit window while maintaining aspect ratio
            display_frame, x_offset, y_offset, disp_w, disp_h = self.resize_frame(
                processed_frame, win_w, win_h)
            
            # Scale UI elements for display
            scale_x = disp_w / self.original_width
            scale_y = disp_h / self.original_height
            
            # Display mode text
            mode_text = "PAUSED" if self.paused else "RUNNING"
            cv2.putText(display_frame, mode_text, 
                       (win_w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if not self.setup_mode:
                self.display_leaderboard(display_frame)
            self.display_instructions(display_frame)
            
            # Show resizable frame
            cv2.imshow('Robot Race Tracker', display_frame)
            
            # Process key presses (same as before)
            key = cv2.waitKey(1) & 0xFF
            # ... (keep your existing key processing logic)
            
            if cv2.getWindowProperty('Robot Race Tracker', cv2.WND_PROP_VISIBLE) < 1:
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = RobotRaceTracker()
    tracker.run()