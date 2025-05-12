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
        self.finish_line_angle = 0  # 0 = horizontal, 90 = vertical
        
        # Expanded robot color ranges in HSV
        self.color_ranges = {
            'red': ([0, 150, 100], [10, 255, 255]),
            'blue': ([100, 150, 100], [130, 255, 255]),
            'green': ([50, 150, 100], [70, 255, 255]),
            'yellow': ([20, 150, 100], [40, 255, 255]),
            'orange': ([10, 150, 100], [20, 255, 255]),
            'purple': ([130, 150, 100], [160, 255, 255]),
            'pink': ([160, 100, 100], [180, 255, 255]),
            'cyan': ([80, 150, 100], [100, 255, 255]),
            'magenta': ([170, 150, 100], [180, 255, 255]),
            'lime': ([40, 150, 100], [50, 255, 255]),
            'teal': ([70, 150, 100], [80, 255, 255]),
            'white': ([0, 0, 200], [180, 50, 255]),
            'black': ([0, 0, 0], [180, 255, 50])
        }
        
        # Tracking variables
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
        """Resize frame while maintaining aspect ratio with letterboxing"""
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
        """Calculate finish line endpoints based on current parameters"""
        if self.finish_line_angle == 0:  # Horizontal line
            return ((self.finish_line_x1, self.finish_line_y), 
                    (self.finish_line_x2, self.finish_line_y))
        elif self.finish_line_angle == 90:  # Vertical line
            return ((self.finish_line_y, self.finish_line_x1), 
                    (self.finish_line_y, self.finish_line_x2))
        else:
            center_x = (self.finish_line_x1 + self.finish_line_x2) // 2
            center_y = self.finish_line_y
            length = abs(self.finish_line_x2 - self.finish_line_x1)
            angle_rad = np.deg2rad(self.finish_line_angle)
            x1 = int(center_x - length/2 * np.cos(angle_rad))
            y1 = int(center_y - length/2 * np.sin(angle_rad))
            x2 = int(center_x + length/2 * np.cos(angle_rad))
            y2 = int(center_y + length/2 * np.sin(angle_rad))
            return ((x1, y1), (x2, y2))
    
    def detect_robots(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        positions = {}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            # Handle red color (which wraps around 0 in HSV)
            if color_name == 'red':
                mask1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                color_mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter out small contours that might be noise
                contours = [c for c in contours if cv2.contourArea(c) > 100]
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Get finish line points
                    line_p1, line_p2 = self.get_finish_line_points()
                    
                    # Only exclude detection if the robot is right on the line
                    if self.finish_line_angle == 0:  # Horizontal line
                        on_line = abs(center_y - line_p1[1]) < 10 and \
                                 line_p1[0] <= center_x <= line_p2[0]
                    else:  # Vertical line
                        on_line = abs(center_x - line_p1[0]) < 10 and \
                                 line_p1[1] <= center_y <= line_p2[1]
                    
                    if not on_line:
                        positions[color_name] = (center_x, center_y)
                        
                        cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                        cv2.putText(frame, color_name, (center_x - 20, center_y - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return positions

    def check_finish_line_crossing(self, positions):
        current_time = time.time() - self.start_time
        line_p1, line_p2 = self.get_finish_line_points()
        
        for color, (x, y) in positions.items():
            if color in self.robot_positions:
                prev_x, prev_y = self.robot_positions[color]
                
                # Check if line was crossed (simple version - could be improved)
                if self.finish_line_angle == 0:  # Horizontal line
                    crossed = (prev_y <= line_p1[1] and y > line_p1[1]) or \
                              (prev_y >= line_p1[1] and y < line_p1[1])
                else:  # Vertical line
                    crossed = (prev_x <= line_p1[0] and x > line_p1[0]) or \
                              (prev_x >= line_p1[0] and x < line_p1[0])
                
                if crossed and not self.has_crossed[color]:
                    self.finish_times[color] = current_time
                    self.has_crossed[color] = True
                    
                    if color not in [r[0] for r in self.leaderboard]:
                        self.leaderboard.append((color, current_time))
                        print(f"{color} robot finished at {current_time:.2f} seconds!")
            
            self.robot_positions[color] = (x, y)
    
    def display_leaderboard(self, frame, scale_x=1.0, scale_y=1.0):
        sorted_leaderboard = sorted(self.leaderboard, key=lambda x: x[1])
        
        base_font_scale = 0.7 * min(scale_x, scale_y)
        line_height = int(30 * scale_y)
        
        cv2.putText(frame, "Leaderboard:", 
                   (int(10 * scale_x), int(30 * scale_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, (0, 0, 255), 2)
        
        for i, (color, t) in enumerate(sorted_leaderboard):
            text = f"{i+1}. {color}: {t:.2f}s"
            cv2.putText(frame, text, 
                       (int(10 * scale_x), int(60 * scale_y + i * line_height)), 
                       cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.85, (0, 255, 0), 2)
    
    def display_instructions(self, frame, scale_x=1.0, scale_y=1.0):
        base_font_scale = 0.5 * min(scale_x, scale_y)
        line_height = int(30 * scale_y)
        
        if self.setup_mode and not self.line_confirmed:
            instructions = [
                "SETUP MODE: Adjust finish line",
                "W/S: Move line up/down",
                "A/D: Move line left/right",
                "Q/E: Rotate line",
                "ENTER: Confirm position",
                "ESC: Cancel confirmation"
            ]
        elif self.setup_mode and self.line_confirmed:
            instructions = [
                "Line position confirmed",
                "ENTER: Start race",
                "ESC: Re-adjust line"
            ]
        else:
            instructions = [
                "SPACE: Pause/Resume",
                "R: Reset race",
                "Q: Quit"
            ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, 
                       (int(10 * scale_x), int(frame.shape[0] - 30 * scale_y - i * line_height)), 
                       cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, (255, 255, 255), 1)
    
    def reset_race(self):
        self.robot_positions = {}
        self.finish_times = {}
        self.has_crossed = defaultdict(bool)
        self.start_time = time.time()
        self.leaderboard = []
        self.race_started = False
        print("Race reset!")
    
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
            
            # Calculate scaling factors for UI elements
            scale_x = disp_w / self.original_width
            scale_y = disp_h / self.original_height
            
            # Display mode text (scaled to window size)
            mode_text = "PAUSED" if self.paused else "RUNNING"
            cv2.putText(display_frame, mode_text, 
                       (win_w - int(150 * scale_x), int(30 * scale_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * min(scale_x, scale_y), (0, 0, 255), 2)
            
            # Display UI elements
            if not self.setup_mode:
                self.display_leaderboard(display_frame, scale_x, scale_y)
            self.display_instructions(display_frame, scale_x, scale_y)
            
            # Show resizable frame
            cv2.imshow('Robot Race Tracker', display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            
            if self.setup_mode:
                if not self.line_confirmed:
                    # Finish line adjustment
                    if key == ord('w'): self.finish_line_y -= 5
                    elif key == ord('s'): self.finish_line_y += 5
                    elif key == ord('a'): 
                        self.finish_line_x1 -= 5
                        self.finish_line_x2 -= 5
                    elif key == ord('d'): 
                        self.finish_line_x1 += 5
                        self.finish_line_x2 += 5
                    elif key == ord('q'): self.finish_line_angle = max(0, self.finish_line_angle - 15)
                    elif key == ord('e'): self.finish_line_angle = min(180, self.finish_line_angle + 15)
                    elif key == 13:  # Enter key
                        self.line_confirmed = True
                else:
                    if key == 13:  # Enter key
                        self.setup_mode = False
                        self.paused = False
                        self.start_time = time.time()
                    elif key == 27:  # ESC key
                        self.line_confirmed = False
            else:
                # Normal operation mode
                if key == ord(' '):  # Space bar
                    self.paused = not self.paused
                    if not self.paused:
                        self.start_time = time.time() - (self.finish_times[max(self.finish_times, key=self.finish_times.get)] if self.finish_times else 0)
                elif key == ord('r'):  # Reset
                    self.reset_race()
                elif key == ord('q'):  # Quit
                    break
            
            if cv2.getWindowProperty('Robot Race Tracker', cv2.WND_PROP_VISIBLE) < 1:
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final results
        print("\nFinal Results:")
        for i, (color, t) in enumerate(sorted(self.leaderboard, key=lambda x: x[1])):
            print(f"{i+1}. {color} robot: {t:.2f} seconds")

if __name__ == "__main__":
    tracker = RobotRaceTracker()
    tracker.run()