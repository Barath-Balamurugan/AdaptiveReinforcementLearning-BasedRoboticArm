import cv2
import time
import numpy as np
from robotic_arm_perception import RoboticArmPerception

class SimulatedRoboticArmPerception(RoboticArmPerception):
    """Simulated version of the perception system for testing RL"""
    
    def __init__(self):
        """Initialize without actual camera hardware"""
        # Skip parent init since we don't need a real camera
        # Initialize simulation environment
        self.arm_position = np.array([0.0, 0.0, 0.0])
        self.arm_velocity = np.array([0.0, 0.0, 0.0])
        self.obstacles = [
            {'id': 'obs1', 'position': [0.3, 0.3, 0.3], 'dimensions': [0.05, 0.05, 0.05]},
            {'id': 'obs2', 'position': [0.2, -0.2, 0.2], 'dimensions': [0.05, 0.05, 0.05]}
        ]
        
        # For visualization
        self.frame_size = (500, 500)
    
    def get_perception_data(self):
        """Generate simulated perception data"""
        # Create simulated arm data
        arm_data = {
            'joints': {
                'base': {'position': [0.0, 0.0, 0.0], 'orientation': [0, 0, 0, 1]},
                'joint1': {'position': [0.0, 0.0, 0.1], 'orientation': [0, 0, 0, 1]},
                'joint2': {'position': [0.1, 0.0, 0.2], 'orientation': [0, 0, 0, 1]}
            },
            'end_effector': {
                'position': self.arm_position.tolist(),
                'orientation': [0, 0, 0, 1],
                'velocity': self.arm_velocity.tolist()
            }
        }
        
        # Create complete perception data
        perception_data = {
            'arm': arm_data,
            'obstacles': self.obstacles,
            'timestamp': time.time()
        }
        
        # Create visualization frame
        visual_frame = self._create_visualization()
        
        return perception_data, visual_frame
    
    def _create_visualization(self):
        """Create a visualization of the simulated environment"""
        # Create blank image
        frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 255
        
        # Calculate center and scale
        center_x, center_y = self.frame_size[0] // 2, self.frame_size[1] // 2
        scale = 200  # pixels per meter
        
        # Draw coordinate axes
        cv2.line(frame, (center_x, center_y), (center_x + 50, center_y), (0, 0, 255), 2)  # X-axis
        cv2.line(frame, (center_x, center_y), (center_x, center_y - 50), (0, 255, 0), 2)  # Y-axis
        
        # Draw arm
        # Base
        cv2.circle(frame, (center_x, center_y), 10, (0, 0, 0), -1)
        
        # Joint 1
        joint1_x = center_x
        joint1_y = center_y - int(0.1 * scale)
        cv2.circle(frame, (joint1_x, joint1_y), 8, (0, 0, 0), -1)
        cv2.line(frame, (center_x, center_y), (joint1_x, joint1_y), (0, 0, 0), 2)
        
        # Joint 2
        joint2_x = center_x + int(0.1 * scale)
        joint2_y = center_y - int(0.2 * scale)
        cv2.circle(frame, (joint2_x, joint2_y), 8, (0, 0, 0), -1)
        cv2.line(frame, (joint1_x, joint1_y), (joint2_x, joint2_y), (0, 0, 0), 2)
        
        # End effector
        ee_x = center_x + int(self.arm_position[0] * scale)
        ee_y = center_y - int(self.arm_position[1] * scale)
        cv2.circle(frame, (ee_x, ee_y), 10, (255, 0, 0), -1)
        cv2.line(frame, (joint2_x, joint2_y), (ee_x, ee_y), (0, 0, 0), 2)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pos = obstacle['position']
            dim = obstacle['dimensions']
            
            # Convert to pixel coordinates
            x = center_x + int(pos[0] * scale)
            y = center_y - int(pos[1] * scale)
            w = int(dim[0] * scale)
            h = int(dim[1] * scale)
            
            # Draw as rectangle
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), -1)
            cv2.putText(frame, obstacle['id'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def update_arm_position(self, position, velocity=[0, 0, 0]):
        """Update the simulated arm position"""
        self.arm_position = np.array(position)
        self.arm_velocity = np.array(velocity)
    
    def update_obstacles(self, obstacles=None):
        """Update the simulated obstacles"""
        if obstacles is not None:
            self.obstacles = obstacles

class SimulatedArmController:
    """Simulated arm controller for testing"""
    
    def __init__(self, perception_system):
        self.perception = perception_system
    
    def move_to_position(self, position):
        """Simulate moving arm to a position"""
        current_pos = self.perception.arm_position
        
        # Check for collisions
        for obstacle in self.perception.obstacles:
            obs_pos = np.array(obstacle['position'])
            obs_dim = np.array(obstacle['dimensions'])
            
            # Simple collision check
            distance = np.linalg.norm(np.array(position) - obs_pos)
            if distance < 0.1:  # Collision threshold
                print(f"Movement failed: collision with {obstacle['id']}")
                return False
        
        # Update position in simulation
        self.perception.update_arm_position(position)
        
        return True
    
    def move_to_home(self):
        """Move to home position"""
        return self.move_to_position([0.0, 0.0, 0.2])
    
    def avoid_nearest_obstacle(self):
        """Generate avoidance motion"""
        current_pos = self.perception.arm_position
        
        # Find nearest obstacle
        nearest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in self.perception.obstacles:
            obs_pos = np.array(obstacle['position'])
            distance = np.linalg.norm(current_pos - obs_pos)
            
            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle
        
        if nearest_obstacle is None:
            return False
        
        # Generate avoidance position
        obs_pos = np.array(nearest_obstacle['position'])
        
        # Vector away from obstacle
        away_vector = current_pos - obs_pos
        away_vector = away_vector / np.linalg.norm(away_vector)
        
        # Move away
        new_position = current_pos + away_vector * 0.1  # 10cm away
        
        return self.move_to_position(new_position.tolist())