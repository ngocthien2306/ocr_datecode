"""
Create sample recipes for testing and demonstration

This script creates 10 different sample recipes with various configurations
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


async def create_10_sample_recipes():
    """Create 10 sample recipes with different configurations"""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    recipe_repo = RecipeRepository(db)
    
    print("Creating indexes...")
    await recipe_repo.create_indexes()
    
    # Default user ID for testing (in production, use real user IDs)
    default_user_id = "admin"
    
    # 10 diverse sample recipes
    sample_recipes = [
        {
            "name": "Coca Cola 330ml Can",
            "product_code": "COCA-330-001",
            "description": "Standard recipe for Coca Cola 330ml cans with high-speed line processing",
            "camera_settings": CameraSettings(
                exposure_time=45.0,
                delay_trigger=80.0,
                gain=1.0,
                brightness=0.5,
                contrast=1.1
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.75,
                recognition_threshold=0.80,
                min_text_size=12,
                max_text_size=180
            ),
            "template_config": {
                "template_type": "beverage_can",
                "roi": [120, 100, 450, 320],
                "rotation_tolerance": 5
            },
            "roi_config": {
                "x": 120,
                "y": 100,
                "width": 450,
                "height": 320
            }
        },
        {
            "name": "Pepsi 500ml Bottle",
            "product_code": "PEPSI-500-002",
            "description": "Recipe for Pepsi 500ml bottles with medium-speed processing",
            "camera_settings": CameraSettings(
                exposure_time=55.0,
                delay_trigger=100.0,
                gain=1.1,
                brightness=0.55,
                contrast=1.0
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.70,
                recognition_threshold=0.75,
                min_text_size=15,
                max_text_size=200
            ),
            "template_config": {
                "template_type": "beverage_bottle",
                "roi": [100, 120, 400, 350],
                "rotation_tolerance": 8
            },
            "roi_config": {
                "x": 100,
                "y": 120,
                "width": 400,
                "height": 350
            }
        },
        {
            "name": "Milk Box 1L Tetra Pack",
            "product_code": "MILK-1L-003",
            "description": "Dairy product 1L tetra pack with precision OCR requirements",
            "camera_settings": CameraSettings(
                exposure_time=60.0,
                delay_trigger=120.0,
                gain=1.2,
                brightness=0.6,
                contrast=1.15
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.82,
                recognition_threshold=0.85,
                min_text_size=18,
                max_text_size=220
            ),
            "template_config": {
                "template_type": "tetra_pack",
                "roi": [150, 130, 480, 380],
                "rotation_tolerance": 3
            },
            "roi_config": {
                "x": 150,
                "y": 130,
                "width": 480,
                "height": 380
            }
        },
        {
            "name": "Snack Chips 100g Package",
            "product_code": "CHIPS-100-004",
            "description": "Small snack package with flexible film, requires special lighting",
            "camera_settings": CameraSettings(
                exposure_time=70.0,
                delay_trigger=90.0,
                gain=1.3,
                brightness=0.65,
                contrast=1.2
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.68,
                recognition_threshold=0.72,
                min_text_size=10,
                max_text_size=160
            ),
            "template_config": {
                "template_type": "flexible_package",
                "roi": [80, 90, 350, 280],
                "rotation_tolerance": 10
            },
            "roi_config": {
                "x": 80,
                "y": 90,
                "width": 350,
                "height": 280
            }
        },
        {
            "name": "Yogurt Cup 150ml",
            "product_code": "YOGURT-150-005",
            "description": "Small dairy cup with curved surface, challenging for OCR",
            "camera_settings": CameraSettings(
                exposure_time=50.0,
                delay_trigger=95.0,
                gain=1.15,
                brightness=0.58,
                contrast=1.12
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.73,
                recognition_threshold=0.78,
                min_text_size=14,
                max_text_size=170
            ),
            "template_config": {
                "template_type": "cup_curved",
                "roi": [110, 105, 380, 300],
                "rotation_tolerance": 6
            },
            "roi_config": {
                "x": 110,
                "y": 105,
                "width": 380,
                "height": 300
            }
        },
        {
            "name": "Beer Can 355ml Premium",
            "product_code": "BEER-355-006",
            "description": "Premium beer can with metallic surface requiring adjusted exposure",
            "camera_settings": CameraSettings(
                exposure_time=40.0,
                delay_trigger=75.0,
                gain=0.95,
                brightness=0.48,
                contrast=1.08
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.77,
                recognition_threshold=0.82,
                min_text_size=13,
                max_text_size=185
            ),
            "template_config": {
                "template_type": "beverage_can_metallic",
                "roi": [125, 95, 440, 310],
                "rotation_tolerance": 4
            },
            "roi_config": {
                "x": 125,
                "y": 95,
                "width": 440,
                "height": 310
            }
        },
        {
            "name": "Instant Noodle Cup 75g",
            "product_code": "NOODLE-75-007",
            "description": "Instant noodle cup with printed date on curved lid",
            "camera_settings": CameraSettings(
                exposure_time=65.0,
                delay_trigger=110.0,
                gain=1.25,
                brightness=0.62,
                contrast=1.18
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.71,
                recognition_threshold=0.76,
                min_text_size=16,
                max_text_size=190
            ),
            "template_config": {
                "template_type": "cup_lid",
                "roi": [130, 115, 420, 340],
                "rotation_tolerance": 7
            },
            "roi_config": {
                "x": 130,
                "y": 115,
                "width": 420,
                "height": 340
            }
        },
        {
            "name": "Water Bottle 1.5L PET",
            "product_code": "WATER-1.5L-008",
            "description": "Large water bottle with transparent PET material",
            "camera_settings": CameraSettings(
                exposure_time=48.0,
                delay_trigger=85.0,
                gain=1.05,
                brightness=0.52,
                contrast=1.05
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.74,
                recognition_threshold=0.79,
                min_text_size=17,
                max_text_size=210
            ),
            "template_config": {
                "template_type": "pet_bottle_large",
                "roi": [105, 110, 460, 360],
                "rotation_tolerance": 9
            },
            "roi_config": {
                "x": 105,
                "y": 110,
                "width": 460,
                "height": 360
            }
        },
        {
            "name": "Energy Drink 250ml Slim Can",
            "product_code": "ENERGY-250-009",
            "description": "Slim energy drink can requiring precise alignment",
            "camera_settings": CameraSettings(
                exposure_time=42.0,
                delay_trigger=78.0,
                gain=1.0,
                brightness=0.5,
                contrast=1.1
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.79,
                recognition_threshold=0.84,
                min_text_size=11,
                max_text_size=175
            ),
            "template_config": {
                "template_type": "slim_can",
                "roi": [115, 98, 410, 295],
                "rotation_tolerance": 3
            },
            "roi_config": {
                "x": 115,
                "y": 98,
                "width": 410,
                "height": 295
            }
        },
        {
            "name": "Juice Box 200ml Kids Pack",
            "product_code": "JUICE-200-010",
            "description": "Small juice box for kids with compact size and bright colors",
            "camera_settings": CameraSettings(
                exposure_time=58.0,
                delay_trigger=105.0,
                gain=1.18,
                brightness=0.57,
                contrast=1.13
            ),
            "model_thresholds": ModelThresholds(
                detection_threshold=0.69,
                recognition_threshold=0.74,
                min_text_size=12,
                max_text_size=165
            ),
            "template_config": {
                "template_type": "small_box",
                "roi": [95, 100, 370, 290],
                "rotation_tolerance": 8
            },
            "roi_config": {
                "x": 95,
                "y": 100,
                "width": 370,
                "height": 290
            }
        }
    ]
    
    print("\n" + "="*80)
    print("Creating 10 Sample Recipes")
    print("="*80 + "\n")
    
    created_count = 0
    failed_count = 0
    
    for idx, recipe_data in enumerate(sample_recipes, 1):
        try:
            recipe = RecipeCreate(**recipe_data)
            created = await recipe_repo.create(recipe, default_user_id)
            created_count += 1
            
            print(f"✅ {idx:2d}. Created: {created.name}")
            print(f"    Product Code: {created.product_code}")
            print(f"    ID: {created.id}")
            print(f"    Camera: Exp={created.camera_settings.exposure_time}ms, "
                  f"Delay={created.camera_settings.delay_trigger}ms")
            print(f"    Thresholds: Detection={created.model_thresholds.detection_threshold}, "
                  f"Recognition={created.model_thresholds.recognition_threshold}")
            print()
            
        except Exception as e:
            failed_count += 1
            print(f"❌ {idx:2d}. Failed: {recipe_data['name']}")
            print(f"    Error: {str(e)}")
            print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successfully created: {created_count} recipes")
    if failed_count > 0:
        print(f"❌ Failed to create: {failed_count} recipes")
    print()
    
    # Display all recipes in database
    print("="*80)
    print("ALL RECIPES IN DATABASE")
    print("="*80 + "\n")
    
    all_recipes = await recipe_repo.get_all(limit=100)
    
    if not all_recipes:
        print("No recipes found in database.")
    else:
        for idx, recipe in enumerate(all_recipes, 1):
            print(f"{idx:2d}. {recipe.name}")
            print(f"    Product Code: {recipe.product_code}")
            print(f"    Camera Settings: Exposure={recipe.camera_settings.exposure_time}ms, "
                  f"Delay={recipe.camera_settings.delay_trigger}ms")
            print(f"    Model Thresholds: Det={recipe.model_thresholds.detection_threshold:.2f}, "
                  f"Rec={recipe.model_thresholds.recognition_threshold:.2f}")
            print(f"    Status: {'Active' if recipe.is_active else 'Inactive'}")
            print(f"    Created: {recipe.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    print(f"Total recipes in database: {len(all_recipes)}")
    print()
    
    # Close connection
    client.close()
    print("✅ Database connection closed")
    print("\n🎉 Sample recipe creation complete!")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║          Sample Recipe Creator for OCR Datecode System        ║
║                                                               ║
║  This script will create 10 diverse sample recipes with      ║
║  different camera settings and model thresholds for testing   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(create_10_sample_recipes())
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
