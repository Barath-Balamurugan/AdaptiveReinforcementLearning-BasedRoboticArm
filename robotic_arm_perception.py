import cv2
import numpy as np
import time
import os
from collections import deque

class RoboticArmPerception:
    def __init__(self, camera_id=0, use_depth=False, calibration_path=None):
        """
        Initialize the camera-based perception system.
        
        Args:
            camera_id: Camera device ID
            use_depth: Whether to use depth sensing (requires RGB-D camera)
            calibration_path: Path to calibration file
        """
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id)
        self.use_depth = use_depth
        
        # For velocity calculation (keeping track of previous positions)
        self.position_history = deque(maxlen=10)
        self.timestamp_history = deque(maxlen=10)
        
        # Load camera calibration if available
        self.camera_matrix = None
        self.dist_coeffs = None
        if calibration_path and os.path.exists(calibration_path):
            calibration = np.load(calibration_path)
            self.camera_matrix = calibration['camera_matrix']
            self.dist_coeffs = calibration['dist_coeffs']
        else:
            print("No calibration file found. Running in uncalibrated mode.")
        
        # Initialize ArUco marker detection for arm tracking
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Marker size in meters
        self.marker_size = 0.1  
        
        # Mapping of marker IDs to arm parts
        self.arm_markers = {
            # 0: "base",
            # 1: "joint1",
            # 2: "joint2",
            0: "end_effector"
        }
        
        # For obstacle detection
        self.obstacle_detector = self._initialize_obstacle_detector()
        
        # Background subtraction for moving obstacle detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False)
        
    def _initialize_obstacle_detector(self):
        """
        Initialize obstacle detection method.
        Currently using simple color-based detection, but could be replaced
        with more advanced methods like YOLO or Mask R-CNN.
        """
        color_ranges = {
            'red': (np.array([0, 120, 70]), np.array([10, 255, 255])),
            'blue': (np.array([100, 100, 100]), np.array([140, 255, 255])),
            'green': (np.array([40, 100, 100]), np.array([80, 255, 255])),
        }
        return color_ranges
    
    def calibrate_camera(self, chessboard_size=(9, 6), num_samples=20):
        """
        Calibrate the camera using a chessboard pattern.
        
        Args:
            chessboard_size: Size of the chessboard (inner corners)
            num_samples: Number of samples to collect for calibration
        """
        print("Starting camera calibration...")
        print("Place the chessboard in view of the camera.")
        print(f"Collecting {num_samples} samples. Press 'c' to capture a sample.")
        
        # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        samples_collected = 0
        while samples_collected < num_samples:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Display the frame
            cv2.imshow('Camera Calibration', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Capture sample
                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                
                if ret:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Draw and display the corners
                    cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
                    cv2.imshow('Camera Calibration', frame)
                    cv2.waitKey(500)  # Display for 500ms
                    
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    
                    samples_collected += 1
                    print(f"Sample {samples_collected}/{num_samples} captured")
                else:
                    print("Chessboard not found. Try again.")
            
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        
        if samples_collected == 0:
            print("No samples collected. Calibration failed.")
            return False
        
        print("Calculating camera calibration...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            
            # Save calibration data
            np.savez('camera_calibration.npz', 
                     camera_matrix=camera_matrix, 
                     dist_coeffs=dist_coeffs)
            
            print("Calibration successful! Saved to camera_calibration.npz")
            return True
        else:
            print("Calibration failed.")
            return False
        
    
    def detect_arm(self, frame):
        """
        Detect and track the robotic arm using ArUco markers.
        
        Args:
            frame: Camera frame
            
        Returns:
            arm_data: Dictionary containing arm joint positions and orientations
        """
        if frame is None:
            return None, None  # Return a tuple with None values
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)
        
        # If no markers detected
        if ids is None or len(ids) == 0:  # Add check for empty IDs
            return None, frame 
        
        arm_data = {
            'joints': {},
            'end_effector': None,
            'timestamp': time.time()
        }
        
        # Draw markers for visualization
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate pose for each marker
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
            
            # Process each detected marker
            for i in range(len(ids)):
                marker_id = ids[i][0]
                
                # Draw axis for visualization
                cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, 
                                rvecs[i], tvecs[i], 0.03)
                
                # Extract position (translation vector)
                position = tvecs[i][0].tolist()
                
                # Convert rotation vector to quaternion
                rotation_matrix, _ = cv2.Rodrigues(rvecs[i][0])
                quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
                
                # Map marker ID to arm part
                if marker_id in self.arm_markers:
                    part_name = self.arm_markers[marker_id]
                    
                    if part_name == "end_effector":
                        arm_data['end_effector'] = {
                            'position': position,
                            'orientation': quaternion.tolist()
                        }
                    else:
                        arm_data['joints'][part_name] = {
                            'position': position,
                            'orientation': quaternion.tolist()
                        }
        else:
            # If no calibration, just use 2D positions
            for i in range(len(ids)):
                marker_id = ids[i][0]
                
                # Calculate center of marker in 2D
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])
                
                position = [center_x, center_y, 0.0]  # Z is unknown
                
                # Map marker ID to arm part
                if marker_id in self.arm_markers:
                    part_name = self.arm_markers[marker_id]
                    
                    if part_name == "end_effector":
                        arm_data['end_effector'] = {
                            'position': position,
                            'orientation': [0, 0, 0, 1]  # Default quaternion
                        }
                    else:
                        arm_data['joints'][part_name] = {
                            'position': position,
                            'orientation': [0, 0, 0, 1]  # Default quaternion
                        }
        
        # Store position history for velocity calculation
        if arm_data.get('end_effector') is not None:
            self.position_history.append(arm_data['end_effector']['position'])
            self.timestamp_history.append(time.time())
            
            # Calculate velocity if we have enough history
            if len(self.position_history) >= 2:
                velocity = self._calculate_velocity()
                # Now it's safe to add velocity
                arm_data['end_effector']['velocity'] = velocity
        
        return arm_data, frame
    
    def _rotation_matrix_to_quaternion(self, rotation_matrix):
        """
        Convert a rotation matrix to quaternion.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            quaternion: [qx, qy, qz, qw]
        """
        trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])

    def _calculate_velocity(self):
        """
        Calculate velocity based on position history.
        
        Returns:
            velocity: [vx, vy, vz]
        """
        if len(self.position_history) < 2:
            return [0, 0, 0]
        
        # Use the most recent positions for more accurate instantaneous velocity
        pos_current = np.array(self.position_history[-1])
        pos_prev = np.array(self.position_history[-2])
        
        time_current = self.timestamp_history[-1]
        time_prev = self.timestamp_history[-2]
        
        time_diff = time_current - time_prev
        
        # Avoid division by zero
        if time_diff < 0.001:
            return [0, 0, 0]
        
        # Calculate velocity
        velocity = (pos_current - pos_prev) / time_diff
        
        # Apply smoothing (optional)
        if len(self.position_history) >= 5:
            # Calculate average velocity over last few positions for smoothing
            velocities = []
            for i in range(len(self.position_history)-1):
                pos_i = np.array(self.position_history[i])
                pos_i_plus_1 = np.array(self.position_history[i+1])
                time_i = self.timestamp_history[i]
                time_i_plus_1 = self.timestamp_history[i+1]
                time_diff_i = time_i_plus_1 - time_i
                if time_diff_i > 0.001:  # Avoid division by zero
                    velocities.append((pos_i_plus_1 - pos_i) / time_diff_i)
            
            if velocities:
                # Average the velocities
                smoothed_velocity = np.mean(velocities, axis=0)
                # Weight current velocity more than the smoothed one
                velocity = 0.7 * velocity + 0.3 * smoothed_velocity
        
        return velocity.tolist()
    
    def detect_obstacles(self, frame):
        """
        Detect obstacles in the frame.
        
        Args:
            frame: Camera frame
            
        Returns:
            obstacles: List of obstacle data
            frame: Annotated frame for visualization
        """
        if frame is None:
            return None, None
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        obstacles = []
        
        # Apply background subtraction for moving object detection
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Detect obstacles using color ranges
        for color_name, (lower, upper) in self.obstacle_detector.items():
            # Create mask for this color
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Combine with foreground mask for moving objects
            # combined_mask = cv2.bitwise_and(color_mask, fg_mask)
            combined_mask = color_mask  # Use only color for now
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Filter out small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                center_x = x + w/2
                center_y = y + h/2
                
                # For depth, we need a depth camera
                # If no depth info, use a placeholder
                depth = 1.0  # Placeholder
                
                if self.use_depth:
                    # If you have depth camera, get real depth
                    # depth = depth_frame[int(center_y), int(center_x)]
                    pass
                
                # Create obstacle data
                obstacle_id = f"{color_name}_{i}"
                obstacle = {
                    'id': obstacle_id,
                    'position': [center_x, center_y, depth],
                    'dimensions': [w, h, w/2],  # Estimate depth as half width
                    'color': color_name
                }
                
                obstacles.append(obstacle)
                
                # Draw bounding box for visualization
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, obstacle_id, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return obstacles, frame
    
    def get_perception_data(self):
        """
        Get complete perception data including arm and obstacles.
        
        Returns:
            perception_data: Dictionary containing all perception information
            visual_frame: Annotated frame for visualization
        """
        # Capture frame
        ret, frame = self.camera.read()
        if not ret or frame is None:
            print("Failed to capture frame")
            return None, None
        
        # Clone frame for visualization
        visual_frame = frame.copy()
        
        # Detect arm
        arm_data, arm_visual = self.detect_arm(frame)
        # Use the returned visual frame even if arm_data is None
        if arm_visual is not None:
            visual_frame = arm_visual
        
        # Detect obstacles
        obstacles, obstacle_visual = self.detect_obstacles(visual_frame)
        if obstacle_visual is not None:
            visual_frame = obstacle_visual
        
        # Create complete perception data
        perception_data = {
            'arm': arm_data,  # This might be None, which is fine
            'obstacles': obstacles if obstacles else [],
            'timestamp': time.time()
        }
        
        return perception_data, visual_frame

    def run(self, visualize=True, save_data=False):
        """
        Run the perception system continuously.
        
        Args:
            visualize: Whether to show visualization
            save_data: Whether to save perception data to file
        """
        data_file = None
        if save_data:
            data_file = open('perception_data.txt', 'w')
        
        try:
            while True:
                try:
                    # Get perception data
                    perception_data, visual_frame = self.get_perception_data()
                    
                    if perception_data is None:
                        print("No perception data available.")
                        time.sleep(0.1)
                        continue
                    
                    # Print detailed data for debugging
                    if (perception_data.get('arm') is not None and 
                        perception_data['arm'] is not None and
                        perception_data['arm'].get('end_effector') is not None):
                        
                        end_effector = perception_data['arm']['end_effector']
                        
                        # Print position if available
                        if 'position' in end_effector:
                            pos = end_effector['position']
                            # print(f"End effector position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                        
                        # Print velocity if available
                        if 'velocity' in end_effector:
                            vel = end_effector['velocity']
                            # print(f"End effector velocity: ({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
                    
                    # Print obstacle info
                    obstacles = perception_data.get('obstacles', [])
                    # print(f"Detected {len(obstacles)} obstacles")
                    
                    # Add information overlay to the visual frame
                    if visualize and visual_frame is not None:
                        # Add data panel at the bottom of the frame
                        h, w = visual_frame.shape[:2]
                        info_panel = np.ones((150, w, 3), dtype=np.uint8) * 240  # Light gray panel
                        
                        # Add text with information
                        cv2.putText(info_panel, "Robotic Arm Perception System", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        # Add position and velocity if available
                        y_pos = 60
                        if (perception_data.get('arm') is not None and 
                            perception_data['arm'] is not None and
                            perception_data['arm'].get('end_effector') is not None):
                            
                            end_effector = perception_data['arm']['end_effector']
                            
                            if 'position' in end_effector:
                                pos = end_effector['position']
                                pos_text = f"Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                                cv2.putText(info_panel, pos_text, (10, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                                y_pos += 25
                            
                            if 'velocity' in end_effector:
                                vel = end_effector['velocity']
                                vel_magnitude = np.linalg.norm(vel)
                                vel_text = f"Velocity: ({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) - Magnitude: {vel_magnitude:.3f}"
                                cv2.putText(info_panel, vel_text, (10, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 1)
                                y_pos += 25
                        
                        # Add obstacle info
                        obstacle_text = f"Obstacles: {len(obstacles)}"
                        cv2.putText(info_panel, obstacle_text, (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 0), 1)
                        
                        # Combine with main frame
                        combined_frame = np.vstack((visual_frame, info_panel))
                        
                        # Show the combined frame
                        cv2.imshow('Robotic Arm Perception', combined_frame)
                        
                        # Exit on 'q' key
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Save data if requested
                    if save_data and data_file is not None:
                        data_file.write(f"{str(perception_data)}\n")
                        
                except Exception as e:
                    print(f"Error in perception loop: {e}")
                    # Print exception traceback for debugging
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)  # Pause before continuing
        
        finally:
            # Clean up
            self.camera.release()
            cv2.destroyAllWindows()
            
            if save_data and data_file is not None:
                data_file.close()

    def format_for_rl(self, perception_data):
        """
        Format perception data for reinforcement learning.
        
        Args:
            perception_data: Raw perception data
            
        Returns:
            state: State vector for RL
        """
        # Initialize state components
        arm_position = [0, 0, 0]
        arm_velocity = [0, 0, 0]
        obstacle_positions = []
        
        # Extract arm data
        if (perception_data['arm'] is not None and 
            perception_data['arm']['end_effector'] is not None):
            arm_position = perception_data['arm']['end_effector']['position']
            
            if 'velocity' in perception_data['arm']['end_effector']:
                arm_velocity = perception_data['arm']['end_effector']['velocity']
        
        # Extract obstacle data (flattened)
        for obstacle in perception_data['obstacles']:
            obstacle_positions.extend(obstacle['position'])
        
        # Pad obstacle data to fixed length if needed
        max_obstacles = 5  # Maximum number of obstacles to consider
        while len(obstacle_positions) < max_obstacles * 3:
            obstacle_positions.extend([0, 0, 0])  # Pad with zeros
        
        # Truncate if too many obstacles
        obstacle_positions = obstacle_positions[:max_obstacles * 3]
        
        # Combine everything into state vector
        state = np.concatenate([
            arm_position,
            arm_velocity,
            obstacle_positions
        ])
        
        return state
    
if __name__ == "__main__":
    # Initialize perception system
    perception = RoboticArmPerception(camera_id=0, use_depth=False)
    
    # Check if calibration file exists, otherwise run calibration
    if not os.path.exists('camera_calibration.npz'):
        print("No calibration file found. Starting calibration procedure...")
        # perception.calibrate_camera()
    
    # Run the perception system
    perception.run(visualize=True, save_data=True)