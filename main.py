import cv2
import numpy as np
import time
from collections import defaultdict

class RobotRaceTracker:
    def __init__(self):
        # Initialize video capture (0 for webcam, or video file path)
        self.cap = cv2.VideoCapture(0)
        
        # Define finish line (horizontal line at y=position)
        self.finish_line_y = 300
        self.finish_line_thickness = 2
        
        # Robot color ranges in HSV (adjust these based on your robot colors)
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
        
    def detect_robots(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        positions = {}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            # Create mask for the color
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (assuming it's the robot)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Get center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                positions[color_name] = (center_x, center_y)
                
                # Draw marker and label
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                cv2.putText(frame, color_name, (center_x - 20, center_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return positions
    
    def check_finish_line(self, positions):
        current_time = time.time() - self.start_time
        
        for color, (x, y) in positions.items():
            # Check if robot has crossed the finish line (moving downward)
            if y > self.finish_line_y and not self.has_crossed[color]:
                if color in self.robot_positions and self.robot_positions[color][1] <= self.finish_line_y:
                    # Robot just crossed the finish line
                    self.finish_times[color] = current_time
                    self.has_crossed[color] = True
                    
                    # Add to leaderboard if not already there
                    if color not in [r[0] for r in self.leaderboard]:
                        self.leaderboard.append((color, current_time))
                        print(f"{color} robot finished at {current_time:.2f} seconds!")
            
            # Update previous position
            self.robot_positions[color] = (x, y)
    
    def display_leaderboard(self, frame):
        # Sort leaderboard by finish time
        sorted_leaderboard = sorted(self.leaderboard, key=lambda x: x[1])
        
        # Display leaderboard
        cv2.putText(frame, "Leaderboard:", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for i, (color, t) in enumerate(sorted_leaderboard):
            text = f"{i+1}. {color}: {t:.2f}s"
            cv2.putText(frame, text, (10, 60 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame horizontally if using webcam
            frame = cv2.flip(frame, 1)
            
            # Draw finish line
            cv2.line(frame, (0, self.finish_line_y), (frame.shape[1], self.finish_line_y), 
                    (0, 0, 255), self.finish_line_thickness)
            
            # Detect robots
            positions = self.detect_robots(frame)
            
            # Check for finish line crossings
            if positions:
                if not self.race_started:
                    self.race_started = True
                    self.start_time = time.time()
                    print("Race started!")
                
                self.check_finish_line(positions)
            
            # Display leaderboard
            self.display_leaderboard(frame)
            
            # Show frame
            cv2.imshow('Robot Race Tracker', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
