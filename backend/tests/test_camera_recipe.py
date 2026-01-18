"""
Test script for camera configuration in recipes
"""
import asyncio
import sys
from app.models.recipe import (
    RecipeCreate, 
    CameraConfiguration, 
    TriggerConfiguration,
    ModelThresholds
)

async def test_camera_recipe_creation():
    """Test creating a recipe with multiple cameras"""
    
    # Create camera configurations
    camera1 = CameraConfiguration(
        camera_id="CAM001",
        model_name="Basler acA1920-40gm",
        serial_number="40123456",
        location="Production Line 1",
        exposure_time=50.0,
        delay_trigger=100.0,
        gain=1.5,
        pixel_format="Mono8",
        trigger_config=TriggerConfiguration(
            trigger_mode=True,
            trigger_source="Line1",
            trigger_selector="FrameStart",
            trigger_activation="RisingEdge"
        )
    )
    
    camera2 = CameraConfiguration(
        camera_id="CAM002",
        model_name="Basler acA1920-40gm",
        serial_number="40123457",
        location="Production Line 1 - Position 2",
        exposure_time=45.0,
        delay_trigger=120.0,
        gain=1.2,
        pixel_format="RGB8",
        trigger_config=TriggerConfiguration(
            trigger_mode=True,
            trigger_source="Software",
            trigger_selector="ExposureStart",
            trigger_activation="RisingEdge"
        )
    )
    
    # Create recipe with multiple cameras
    recipe = RecipeCreate(
        name="Multi-Camera Test Recipe",
        product_code="PROD-MULTI-001",
        description="Test recipe with multiple camera configurations",
        delay_reject=150.0,
        cameras=[camera1, camera2],
        model_thresholds=ModelThresholds(
            detection_threshold=0.6,
            recognition_threshold=0.7,
            min_text_size=12,
            max_text_size=180
        ),
        is_active=True
    )
    
    # Print recipe data
    print("Recipe created successfully!")
    print(f"Recipe Name: {recipe.name}")
    print(f"Product Code: {recipe.product_code}")
    print(f"Delay Reject: {recipe.delay_reject} ms")
    print(f"Number of Cameras: {len(recipe.cameras)}")
    print("\nCamera Configurations:")
    for i, cam in enumerate(recipe.cameras, 1):
        print(f"\n  Camera {i}:")
        print(f"    ID: {cam.camera_id}")
        print(f"    Model: {cam.model_name}")
        print(f"    Serial: {cam.serial_number}")
        print(f"    Location: {cam.location}")
        print(f"    Exposure Time: {cam.exposure_time} ms")
        print(f"    Delay Trigger: {cam.delay_trigger} ms")
        print(f"    Gain: {cam.gain}")
        print(f"    Pixel Format: {cam.pixel_format}")
        print(f"    Trigger Mode: {cam.trigger_config.trigger_mode}")
        print(f"    Trigger Source: {cam.trigger_config.trigger_source}")
        print(f"    Trigger Selector: {cam.trigger_config.trigger_selector}")
        print(f"    Trigger Activation: {cam.trigger_config.trigger_activation}")
    
    # Convert to dict to verify serialization
    recipe_dict = recipe.model_dump()
    print("\n✅ Recipe serialization successful!")
    print(f"   Cameras in dict: {len(recipe_dict['cameras'])}")
    
    return recipe

if __name__ == "__main__":
    print("Testing Camera Recipe Configuration...")
    print("=" * 60)
    try:
        asyncio.run(test_camera_recipe_creation())
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
