import cv2 

from robotic_arm_perception import RoboticArmPerception

def test_components():
    """Test individual components of the perception system."""
    perception = RoboticArmPerception(camera_id=0)
    
    # Test arm detection
    print("Testing arm detection...")
    ret, frame = perception.camera.read()
    arm_data, visual_frame = perception.detect_arm(frame)
    if arm_data is not None:
        print("Arm detected!")
        print(f"End effector: {arm_data['end_effector']}")
        cv2.imshow('Arm Detection', visual_frame)
        cv2.waitKey(0)
    else:
        print("No arm detected.")
    
    # Test obstacle detection
    print("Testing obstacle detection...")
    ret, frame = perception.camera.read()
    obstacles, visual_frame = perception.detect_obstacles(frame)
    print(f"Detected {len(obstacles)} obstacles")
    cv2.imshow('Obstacle Detection', visual_frame)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment to test individual components
    test_components()
    
    # Or run the full system
    # perception = RoboticArmPerception(camera_id=0)
    # perception.run()