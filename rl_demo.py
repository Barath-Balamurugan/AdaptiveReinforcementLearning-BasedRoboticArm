import cv2

from simulation import SimulatedArmController, SimulatedRoboticArmPerception
from robotic_arm_perception import RoboticArmPerception
from robotic_arm_env import RoboticArmEnv

def run_rl_demo(use_simulation=True):
    """Run a simplified RL demo"""
    print("Starting RL demo...")
    
    # Initialize perception and arm controller
    if use_simulation:
        perception = SimulatedRoboticArmPerception()
        arm_controller = SimulatedArmController(perception)
    else:
        perception = RoboticArmPerception(camera_id=0)
        # Replace with your actual arm controller
        # arm_controller = YourRealArmController()
    
    # Initialize environment
    env = RoboticArmEnv(perception, arm_controller)
    
    # Run a few episodes manually
    for episode in range(3):
        print(f"\nEpisode {episode+1}")
        state = env.reset()
        
        done = False
        total_reward = 0
        
        step = 0
        while not done and step < 40:  # Max 20 steps
            # For demo, alternate between actions
            action = step % 2
            
            print(f"Step {step+1}, taking action: {action}")
            next_state, reward, done, info = env.step(action)
            
            print(f"Reward: {reward:.2f}, Distance to goal: {info['distance_to_goal']:.2f}")
            
            state = next_state
            total_reward += reward
            step += 1
            
            # Show visualization
            perception_data, visual_frame = perception.get_perception_data()
            if visual_frame is not None:
                cv2.imshow('RL Demo', visual_frame)
                cv2.waitKey(500)  # Slow down for visualization
        
        print(f"Episode complete. Total reward: {total_reward:.2f}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run with simulation by default
    run_rl_demo(use_simulation=True)