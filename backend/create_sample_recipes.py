"""
Script to create sample recipes for testing

This script demonstrates how to create recipes with different configurations
for the OCR datecode system.
"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.repositories.recipe_repository import RecipeRepository
from app.models.recipe import RecipeCreate, CameraSettings, ModelThresholds


async def create_sample_recipes():
    """Create sample recipes in the database"""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    recipe_repo = RecipeRepository(db)
    
    print("Creating indexes...")
    await recipe_repo.create_indexes()
    
    # Sample recipes
    sample_recipes = [
        {
            "name": "Product A - Standard",
            "product_code": "PROD-A-001",
            "description": "Standard recipe for Product A with default settings",
            "camera_settings": CameraSettings(
                exposure_time=50.0,
                delay_trigger=100.0,
                gain=1.0,
                brightness=0.5,
                contrast=1.0
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.6,
                recognition_threshold=0.7,
                min_text_size=10,
                max_text_size=200
            ),
            "template_config": {
                "template_type": "standard",
                "roi": [100, 100, 400, 300]
            },
            "roi_config": {
                "x": 100,
                "y": 100,
                "width": 400,
                "height": 300
            }
        },
        {
            "name": "Product B - High Precision",
            "product_code": "PROD-B-002",
            "description": "High precision recipe for Product B with optimized settings",
            "camera_settings": CameraSettings(
                exposure_time=75.0,
                delay_trigger=150.0,
                gain=1.2,
                brightness=0.6,
                contrast=1.1
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.8,
                recognition_threshold=0.85,
                min_text_size=15,
                max_text_size=250
            ),
            "template_config": {
                "template_type": "precise",
                "roi": [150, 150, 500, 400]
            },
            "roi_config": {
                "x": 150,
                "y": 150,
                "width": 500,
                "height": 400
            }
        },
        {
            "name": "Product C - Fast Processing",
            "product_code": "PROD-C-003",
            "description": "Fast processing recipe for Product C",
            "camera_settings": CameraSettings(
                exposure_time=30.0,
                delay_trigger=50.0,
                gain=0.8,
                brightness=0.4
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.5,
                recognition_threshold=0.6,
                min_text_size=8,
                max_text_size=180
            ),
            "template_config": {
                "template_type": "fast",
                "roi": [80, 80, 350, 250]
            },
            "roi_config": {
                "x": 80,
                "y": 80,
                "width": 350,
                "height": 250
            }
        }
    ]
    
    # Create a default admin user ID for testing
    # In production, this should be a real user ID from the users collection
    default_user_id = "admin_test_user"
    
    print("\nCreating sample recipes...")
    for recipe_data in sample_recipes:
        try:
            recipe = RecipeCreate(**recipe_data)
            created = await recipe_repo.create(recipe, default_user_id)
            print(f"✅ Created recipe: {created.name} (ID: {created.id})")
        except Exception as e:
            print(f"❌ Failed to create recipe '{recipe_data['name']}': {str(e)}")
    
    # Show all recipes
    print("\n" + "="*60)
    print("All recipes in database:")
    print("="*60)
    
    all_recipes = await recipe_repo.get_all()
    for idx, recipe in enumerate(all_recipes, 1):
        print(f"\n{idx}. {recipe.name}")
        print(f"   Product Code: {recipe.product_code}")
        print(f"   Description: {recipe.description}")
        print(f"   Camera Settings:")
        print(f"     - Exposure Time: {recipe.camera_settings.exposure_time}ms")
        print(f"     - Delay Trigger: {recipe.camera_settings.delay_trigger}ms")
        print(f"   Model Thresholds:")
        print(f"     - Detection: {recipe.model_thresholds.detection_threshold}")
        print(f"     - Recognition: {recipe.model_thresholds.recognition_threshold}")
        print(f"   Created: {recipe.created_at}")
        print(f"   ID: {recipe.id}")
    
    print(f"\n✅ Total recipes: {len(all_recipes)}")
    
    # Close connection
    client.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(create_sample_recipes())
