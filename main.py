import cv2
import numpy as np
import time
from collections import defaultdict
import os  # Added for Docker compatibility

class RobotRaceTracker:
    def __init__(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Finish line parameters (will be adjustable)
        self.finish_line_y = 300
        self.finish_line_x1 = 100
        self.finish_line_x2 = 500
        self.finish_line_thickness = 2
        self.finish_line_angle = 0  # 0 = horizontal, 90 = vertical
        
        # Robot color ranges in HSV
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255])
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
        
    def get_finish_line_points(self):
        """Calculate finish line endpoints based on current parameters"""
        if self.finish_line_angle == 0:  # Horizontal line
            return ((self.finish_line_x1, self.finish_line_y), 
                    (self.finish_line_x2, self.finish_line_y))
        elif self.finish_line_angle == 90:  # Vertical line
            return ((self.finish_line_y, self.finish_line_x1), 
                    (self.finish_line_y, self.finish_line_x2))
        else:
            # For angled lines (not fully implemented)
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
        
        # Create a mask to exclude the finish line area
        mask_exclude_line = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        line_p1, line_p2 = self.get_finish_line_points()
        
        # Create a thicker line for masking (to exclude area around the line)
        line_thickness = self.finish_line_thickness + 20  # Add buffer around the line
        cv2.line(mask_exclude_line, line_p1, line_p2, 0, line_thickness)
        
        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Combine with our exclusion mask
            combined_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask_exclude_line)
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x = x + w // 2
                center_y = y + h // 2
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
    
    def display_leaderboard(self, frame):
        sorted_leaderboard = sorted(self.leaderboard, key=lambda x: x[1])
        
        cv2.putText(frame, "Leaderboard:", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for i, (color, t) in enumerate(sorted_leaderboard):
            text = f"{i+1}. {color}: {t:.2f}s"
            cv2.putText(frame, text, (10, 60 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def display_instructions(self, frame):
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
            cv2.putText(frame, text, (10, frame.shape[0] - 30 - i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
            
            # Draw finish line
            line_p1, line_p2 = self.get_finish_line_points()
            line_color = (0, 0, 255) if not self.line_confirmed else (0, 255, 0)
            cv2.line(frame, line_p1, line_p2, line_color, self.finish_line_thickness)
            
            # Display current mode
            mode_text = "PAUSED" if self.paused else "RUNNING"
            cv2.putText(frame, mode_text, (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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
            
            if not self.paused and not self.setup_mode:
                # Detect robots and check finish line
                positions = self.detect_robots(frame)
                if positions:
                    if not self.race_started:
                        self.race_started = True
                        print("Race started!")
                    self.check_finish_line_crossing(positions)
            
            # Display leaderboard and instructions
            if not self.setup_mode:
                self.display_leaderboard(frame)
            self.display_instructions(frame)
            
            # Show frame (with Docker compatibility check)
            if os.environ.get('DISPLAY'):
                cv2.imshow('Robot Race Tracker', frame)
            else:
                # Running headless, just process frames without showing
                pass
                
        self.cap.release()
        if os.environ.get('DISPLAY'):
            cv2.destroyAllWindows()
        
        # Print final results
        print("\nFinal Results:")
        for i, (color, t) in enumerate(sorted(self.leaderboard, key=lambda x: x[1])):
            print(f"{i+1}. {color} robot: {t:.2f} seconds")

if __name__ == "__main__":
    # Try to use the first available camera (0) or a video source
    tracker = RobotRaceTracker()
    
    # For Docker, you might want to use a video file instead:
    # tracker.cap = cv2.VideoCapture('race.mp4')
    
    tracker.run()