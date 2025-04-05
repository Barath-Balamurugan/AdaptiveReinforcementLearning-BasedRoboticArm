import gym
from gym import spaces
import numpy as np
import time

class RoboticArmEnv(gym.Env):
    """Robotic Arm Environment for Reinforcement Learning"""
    
    def __init__(self, perception_system, arm_controller):
        super(RoboticArmEnv, self).__init__()
        
        # Initialize perception system
        self.perception = perception_system
        
        # Initialize arm controller
        self.arm_controller = arm_controller
        
        # Define action space (macro-actions)
        self.action_space = spaces.Discrete(2)  # 0: MoveToWaypoint, 1: AvoidObstacle
        
        # Parameters for waypoint generation
        self.step_size = 0.05  # 5cm steps
        
        # Define observation space
        # [arm_position (3), arm_velocity (3), obstacle_positions (5*3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        
        # Set goal position
        self.goal_position = np.array([0.5, 0.5, 0.5])  # Example goal
        
        # Tracking previous states for SMDP
        self.current_option = None
        self.option_start_time = None
    
    def reset(self):
        """Reset the environment"""
        # Move arm to initial position
        self.arm_controller.move_to_home()
        
        # Terminate any ongoing option
        self.current_option = None
        
        # Get initial state
        perception_data, _ = self.perception.get_perception_data()
        state = self.perception.format_for_rl(perception_data)
        
        return state
    
    def step(self, action):
        """
        Execute action and return next state, reward, done, info
        
        Args:
            action: Action index (0: MoveToWaypoint, 1: AvoidObstacle)
            
        Returns:
            next_state: State after action execution
            reward: Reward received
            done: Whether episode has ended
            info: Additional information
        """
        # Record start time if starting new option
        if self.current_option != action:
            self.current_option = action
            self.option_start_time = time.time()
        
        # Execute action based on type
        if action == 0:  # MoveToWaypoint
            # Generate waypoint toward goal
            waypoint = self._generate_waypoint()
            success = self.arm_controller.move_to_position(waypoint)
        else:  # AvoidObstacle
            success = self.arm_controller.avoid_nearest_obstacle()
        
        # Get new state
        perception_data, _ = self.perception.get_perception_data()
        next_state = self.perception.format_for_rl(perception_data)
        
        # Extract arm position
        arm_position = perception_data['arm']['end_effector']['position'] if (
            perception_data['arm'] is not None and 
            perception_data['arm']['end_effector'] is not None) else [0, 0, 0]
        
        # Calculate reward
        reward = self._compute_reward(perception_data)
        
        # Check if done
        done = self._check_done(arm_position)
        
        # Calculate option duration
        option_duration = time.time() - self.option_start_time
        
        # Additional info
        info = {
            'success': success,
            'distance_to_goal': np.linalg.norm(np.array(arm_position) - self.goal_position),
            'option_duration': option_duration
        }
        
        return next_state, reward, done, info
    
    def _generate_waypoint(self):
        """Generate waypoint toward goal"""
        # Get current position
        perception_data, _ = self.perception.get_perception_data()
        
        if (perception_data['arm'] is None or 
            perception_data['arm']['end_effector'] is None):
            # Default waypoint if no perception data
            return [0.2, 0.2, 0.2]
        
        current_position = np.array(perception_data['arm']['end_effector']['position'])
        
        # Move toward goal
        direction = self.goal_position - current_position
        distance = np.linalg.norm(direction)
        
        if distance < 0.001:
            return current_position.tolist()
        
        # Normalize direction and scale by step size
        direction = direction / distance
        step = min(self.step_size, distance)  # Don't overshoot
        
        waypoint = current_position + step * direction
        return waypoint.tolist()
    
    def _compute_reward(self, perception_data):
        """Compute reward based on perception data"""
        # Default reward for failure cases
        if perception_data['arm'] is None or perception_data['arm']['end_effector'] is None:
            return -10.0
        
        # Extract current position
        current_position = np.array(perception_data['arm']['end_effector']['position'])
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(current_position - self.goal_position)
        
        # Base reward based on distance (negative to encourage minimizing distance)
        reward = -distance_to_goal * 10.0
        
        # Reward for reaching goal
        if distance_to_goal < 0.05:
            reward += 100.0
        
        # Penalty for collisions or near-collisions
        for obstacle in perception_data['obstacles']:
            obstacle_position = np.array(obstacle['position'])
            distance_to_obstacle = np.linalg.norm(current_position - obstacle_position)
            
            # Collision penalty
            if distance_to_obstacle < 0.05:  # Collision threshold
                reward -= 100.0
            # Near-collision penalty (soft boundary)
            elif distance_to_obstacle < 0.15:  # Safety threshold
                penalty = 50.0 * (0.15 - distance_to_obstacle) / 0.15
                reward -= penalty
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def _check_done(self, arm_position):
        """Check if episode is done"""
        # Done if goal reached
        distance_to_goal = np.linalg.norm(np.array(arm_position) - self.goal_position)
        if distance_to_goal < 0.05:
            return True
        
        # Check if perception_data contains collision information
        # This would need to be added to your perception system
        
        return False