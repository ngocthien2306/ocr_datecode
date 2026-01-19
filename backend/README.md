# OCR Datecode Backend API

FastAPI backend with Clean Architecture for OCR Datecode system.

## 🎯 Features

### ✅ **Recipe Management System** (NEW!)

Complete receipt management with role-based permissions:

- **New Receipt**: Create recipes for machine training
- **Load Receipt**: Load recipe to verify datecode
- **Save Receipt**: Update existing recipe configurations
- **Delete Receipt**: Remove obsolete recipes (Admin only)

Each recipe includes:

- 📸 **Camera Settings**: Exposure time, delay trigger, gain, brightness, contrast
- 🤖 **Model Thresholds**: Detection/recognition thresholds for AI models
- 📐 **ROI Config**: Region of interest configuration
- 🎯 **Template Config**: Template matching settings

### ✅ **Three-Tier Permission System**

- **👤 Operator**: Load recipes, input datecode for verification
- **👨‍💼 Supervisor**: Create/edit recipes + all Operator permissions
- **👑 Admin**: Full access + user management + delete recipes

### ✅ **Authentication & Authorization**

- JWT-based authentication
- Role-based access control (RBAC)
- Password reset functionality

### ✅ **User Management**

- User CRUD operations
- Role assignment
- Profile management

## Tech Stack

- **FastAPI**: Modern, fast web framework
- **MongoDB**: NoSQL database with Motor async driver
- **JWT**: Secure token-based authentication
- **Pydantic**: Data validation
- **Clean Architecture**: Separation of concerns

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── endpoints/      # API route handlers
│   │   │   ├── auth.py
│   │   │   ├── users.py
│   │   │   └── recipes.py
│   │   └── dependencies/   # Dependency injection
│   │       └── auth.py
│   ├── core/              # Core configuration
│   │   ├── config.py
│   │   └── security.py
│   ├── db/                # Database connection
│   │   └── mongodb.py
│   ├── models/            # Pydantic models
│   │   ├── user.py
│   │   └── recipe.py
│   ├── repositories/      # Data access layer
│   │   ├── user_repository.py
│   │   └── recipe_repository.py
│   ├── schemas/           # Request/Response schemas
│   │   └── auth.py
│   └── main.py           # FastAPI application
├── init_db.py            # Database initialization script
├── requirements.txt
└── .env
```

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=ocr_datecode_db
SECRET_KEY=your-secret-key-change-this
```

### 3. Start MongoDB

Make sure MongoDB is running:

```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or install MongoDB locally
mongod
```

### 4. Initialize Database

Create default users (admin, supervisor, operator):

```bash
python init_db.py
```

This creates:

- Admin: `admin` / `admin123`
- Supervisor: `supervisor` / `supervisor123`
- Operator: `operator` / `operator123`

### 5. Create Sample Recipes (Optional)

```bash
python create_sample_recipes.py
```

This creates 3 sample recipes for testing.

### 6. Run API Server

```bash
uvicorn app.main:app --reload
```

API will be available at: http://localhost:8000

- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 Documentation

- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete implementation overview
- **[RECIPE_API_DOCS.md](RECIPE_API_DOCS.md)** - Comprehensive API documentation (English)
- **[RECIPE_DOCS_VI.md](RECIPE_DOCS_VI.md)** - Hướng dẫn sử dụng (Tiếng Việt)

## API Endpoints

### Authentication

- `POST /api/auth/login` - Login and get JWT token

### Users (Admin only)

- `POST /api/users/` - Create new user
- `GET /api/users/` - List all users
- `GET /api/users/me` - Get current user info
- `GET /api/users/{id}` - Get user by ID
- `PUT /api/users/{id}` - Update user
- `DELETE /api/users/{id}` - Delete user
- `POST /api/users/change-password` - Change own password
- `POST /api/users/{id}/reset-password` - Reset user password (Admin)

### Recipes (NEW!)

- `POST /api/recipes/` - Create recipe (Supervisor+)
- `GET /api/recipes/` - List all recipes (All users)
- `GET /api/recipes/{id}` - Get recipe by ID (All users)
- `PUT /api/recipes/{id}` - Update recipe (Supervisor+)
- `DELETE /api/recipes/{id}` - Delete recipe (Admin only)
- `GET /api/recipes/search` - Search recipes (All users)
- `GET /api/recipes/stats/count` - Get recipe count (All users)
- `GET /api/recipes/` - List all recipes (All)
- `GET /api/recipes/{id}` - Get recipe by ID (All)
- `GET /api/recipes/name/{name}` - Get recipe by name (All)
- `PUT /api/recipes/{id}` - Update recipe (Supervisor+)
- `DELETE /api/recipes/{id}` - Delete recipe (Supervisor+)
- `POST /api/recipes/validate-datecode` - Validate datecode (All)

## Role Permissions

| Feature           | Operator | Supervisor | Admin |
| ----------------- | -------- | ---------- | ----- |
| Login             | ✅       | ✅         | ✅    |
| Load Recipe       | ✅       | ✅         | ✅    |
| Validate Datecode | ✅       | ✅         | ✅    |
| Create Recipe     | ❌       | ✅         | ✅    |
| Edit Recipe       | ❌       | ✅         | ✅    |
| Delete Recipe     | ❌       | ✅         | ✅    |
| View Users        | ❌       | ✅         | ✅    |
| Create User       | ❌       | ❌         | ✅    |
| Edit User         | ❌       | ❌         | ✅    |
| Delete User       | ❌       | ❌         | ✅    |
| Reset Password    | ❌       | ❌         | ✅    |

## Example Usage

### 1. Login

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user": {
    "username": "admin",
    "email": "admin@suntech.com",
    "role": "admin"
  }
}
```

### 2. Create Recipe (with token)

```bash
curl -X POST "http://localhost:8000/api/recipes/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Product A Recipe",
    "description": "Recipe for Product A datecode detection",
    "camera_settings": {
      "exposure_time": 5.0,
      "delay_trigger": 100.0,
      "gain": 1.5,
      "resolution": "1920x1080"
    },
    "model_thresholds": [
      {
        "model_name": "datecode_detection_v1",
        "confidence_threshold": 0.85,
        "iou_threshold": 0.5
      }
    ],
    "datecode_pattern": "^\\d{8}$"
  }'
```

### 3. Validate Datecode

```bash
curl -X POST "http://localhost:8000/api/recipes/validate-datecode" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "recipe_id": "RECIPE_ID",
    "datecode_input": "20240115"
  }'
```

## Development

Run with auto-reload:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Production

For production, use proper secret key and deployment configuration:

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Run with gunicorn + uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Testing

Access Swagger UI for interactive API testing:
http://localhost:8000/docs

## License

Suntech Automation © 2024
