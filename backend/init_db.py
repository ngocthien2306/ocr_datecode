"""
Initialize database with default admin user
Run this script once to setup the initial admin account
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.models.user import UserCreate, UserRole
from app.repositories.user_repository import UserRepository
from datetime import datetime


async def init_database():
    """Initialize database with default admin user"""
    print("🔄 Connecting to MongoDB...")
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]

    print("🔄 Initializing User Repository...")
    user_repo = UserRepository(db)

    # Create indexes
    await user_repo.create_indexes()
    print("✅ Database indexes created")

    # Check if admin already exists
    existing_admin = await user_repo.get_user_by_username("admin")

    if existing_admin:
        print("⚠️  Admin user already exists!")
        print(f"   Username: {existing_admin.username}")
        print(f"   Email: {existing_admin.email}")
        print(f"   Role: {existing_admin.role}")
    else:
        # Create default admin user
        print("🔄 Creating default admin user...")
        admin_user = UserCreate(
            username="admin",
            email="admin@suntech.com",
            full_name="System Administrator",
            password="admin123",  # Change this in production!
            role=UserRole.ADMIN,
            is_active=True
        )

        created_user = await user_repo.create_user(admin_user)
        print("✅ Admin user created successfully!")
        print(f"   Username: {created_user.username}")
        print(f"   Email: {created_user.email}")
        print(f"   Role: {created_user.role}")
        print(f"   Password: admin123")
        print("\n⚠️  IMPORTANT: Change the admin password after first login!")

    # Create sample operator user
    existing_operator = await user_repo.get_user_by_username("operator")
    if not existing_operator:
        print("\n🔄 Creating sample operator user...")
        operator_user = UserCreate(
            username="operator",
            email="operator@suntech.com",
            full_name="Operator User",
            password="operator123",
            role=UserRole.OPERATOR,
            is_active=True
        )

        created_operator = await user_repo.create_user(operator_user)
        print("✅ Operator user created successfully!")
        print(f"   Username: {created_operator.username}")
        print(f"   Password: operator123")

    # Create sample supervisor user
    existing_supervisor = await user_repo.get_user_by_username("supervisor")
    if not existing_supervisor:
        print("\n🔄 Creating sample supervisor user...")
        supervisor_user = UserCreate(
            username="supervisor",
            email="supervisor@suntech.com",
            full_name="Supervisor User",
            password="supervisor123",
            role=UserRole.SUPERVISOR,
            is_active=True
        )

        created_supervisor = await user_repo.create_user(supervisor_user)
        print("✅ Supervisor user created successfully!")
        print(f"   Username: {created_supervisor.username}")
        print(f"   Password: supervisor123")

    print("\n✅ Database initialization completed!")
    print("\n📋 Summary:")
    print("   - Admin user: admin / admin123")
    print("   - Supervisor user: supervisor / supervisor123")
    print("   - Operator user: operator / operator123")
    print("\n🚀 You can now start the API server with: uvicorn app.main:app --reload")

    client.close()


if __name__ == "__main__":
    asyncio.run(init_database())
