# Recipe Management API Documentation

## Overview

The Recipe Management system provides functionality to manage OCR recipes (receipts) with role-based access control. This document explains the implementation according to the requirements.

## Requirements Implementation

### 1. Receipt Management Functions ✅

The software must have the following receipt management capabilities:

- ✅ **New Receipt**: Create new recipes with camera settings and model thresholds
- ✅ **Load Receipt**: Load existing recipes from the database
- ✅ **Save Receipt**: Update existing recipes with new configurations
- ✅ **Delete Receipt**: Permanently remove recipes (Admin only)

### 2. Three-Tier Permission System ✅

#### **Operator** (Lowest Level)
- Can **load receipts** only
- Can input datecode fields to verify with read content
- **Cannot** create, edit, or delete recipes
- Access: `GET /api/recipes/`, `GET /api/recipes/{id}`

#### **Supervisor** (Middle Level)
- Can **create new receipts** (New Receipt)
- Can **edit old receipt content** (Save Receipt)
- Has all Operator permissions
- Access: `POST /api/recipes/`, `PUT /api/recipes/{id}`, plus all Operator endpoints

#### **Admin** (Highest Level)
- Has **all permissions**
- Can **change/reset passwords** for Operator and Supervisor
- Can **delete receipts** permanently
- Access: All endpoints including `DELETE /api/recipes/{id}`

### 3. Recipe Features ✅

Each recipe includes:

#### Camera Settings
- ✅ **Exposure Time**: Camera exposure time in milliseconds
- ✅ **Delay Trigger**: Trigger delay in milliseconds
- ✅ Additional settings: Gain, Brightness, Contrast

#### Model Thresholds
- ✅ **Detection Threshold**: For trained detection models (0.0 - 1.0)
- ✅ **Recognition Threshold**: For trained recognition models (0.0 - 1.0)
- ✅ Min/Max text size constraints

#### Additional Configuration
- ✅ **Template Config**: Template matching configuration
- ✅ **ROI Config**: Region of interest configuration
- ✅ **Product Code**: Unique identifier for products
- ✅ Audit trail: Created/updated by, timestamps

---

## API Endpoints

### Authentication Required
All recipe endpoints require authentication via JWT Bearer token.

### 1. Create Recipe (New Receipt)

**Endpoint**: `POST /api/recipes/`  
**Permission**: Supervisor, Admin  
**Description**: Create a new recipe for machine training or new production

```json
{
  "name": "Product A - Standard",
  "product_code": "PROD-A-001",
  "description": "Standard recipe for Product A",
  "camera_settings": {
    "exposure_time": 50.0,
    "delay_trigger": 100.0,
    "gain": 1.0,
    "brightness": 0.5,
    "contrast": 1.0
  },
  "model_thresholds": {
    "detection_threshold": 0.6,
    "recognition_threshold": 0.7,
    "min_text_size": 10,
    "max_text_size": 200
  },
  "template_config": {
    "template_type": "standard",
    "roi": [100, 100, 400, 300]
  },
  "roi_config": {
    "x": 100,
    "y": 100,
    "width": 400,
    "height": 300
  },
  "is_active": true
}
```

**Response**: `201 Created`
```json
{
  "id": "507f1f77bcf86cd799439011",
  "name": "Product A - Standard",
  "product_code": "PROD-A-001",
  "created_by": "user_id",
  "updated_by": "user_id",
  "created_at": "2025-11-30T10:00:00",
  "updated_at": "2025-11-30T10:00:00",
  ...
}
```

---

### 2. Load Recipes (List All)

**Endpoint**: `GET /api/recipes/`  
**Permission**: Operator, Supervisor, Admin  
**Description**: Load all recipes with pagination

**Query Parameters**:
- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum records to return (default: 100, max: 100)
- `is_active`: Filter by active status (optional)

**Response**: `200 OK`
```json
[
  {
    "id": "507f1f77bcf86cd799439011",
    "name": "Product A - Standard",
    "product_code": "PROD-A-001",
    "camera_settings": {...},
    "model_thresholds": {...},
    ...
  }
]
```

---

### 3. Load Single Recipe

**Endpoint**: `GET /api/recipes/{recipe_id}`  
**Permission**: Operator, Supervisor, Admin  
**Description**: Load a specific recipe to verify datecode with read content

**Response**: `200 OK`
```json
{
  "id": "507f1f77bcf86cd799439011",
  "name": "Product A - Standard",
  "product_code": "PROD-A-001",
  "camera_settings": {
    "exposure_time": 50.0,
    "delay_trigger": 100.0
  },
  "model_thresholds": {
    "detection_threshold": 0.6,
    "recognition_threshold": 0.7
  },
  ...
}
```

---

### 4. Save Recipe (Update)

**Endpoint**: `PUT /api/recipes/{recipe_id}`  
**Permission**: Supervisor, Admin  
**Description**: Update existing recipe content

**Request Body** (all fields optional):
```json
{
  "name": "Product A - Updated",
  "camera_settings": {
    "exposure_time": 60.0,
    "delay_trigger": 120.0
  },
  "model_thresholds": {
    "detection_threshold": 0.7,
    "recognition_threshold": 0.75
  }
}
```

**Response**: `200 OK`

---

### 5. Delete Recipe

**Endpoint**: `DELETE /api/recipes/{recipe_id}`  
**Permission**: Admin only  
**Description**: Permanently delete a recipe

**Response**: `204 No Content`

---

### 6. Search Recipes

**Endpoint**: `GET /api/recipes/search?q={query}`  
**Permission**: Operator, Supervisor, Admin  
**Description**: Search recipes by name, product code, or description

**Query Parameters**:
- `q`: Search query (required)
- `skip`: Pagination offset (default: 0)
- `limit`: Records per page (default: 100)

---

### 7. Get Recipe Count

**Endpoint**: `GET /api/recipes/stats/count`  
**Permission**: Operator, Supervisor, Admin  
**Description**: Get total count of recipes

**Query Parameters**:
- `is_active`: Filter by active status (optional)

**Response**: `200 OK`
```json
{
  "count": 42
}
```

---

## Data Models

### CameraSettings
```python
{
  "exposure_time": float,      # milliseconds (required)
  "delay_trigger": float,       # milliseconds (required)
  "gain": float,                # optional
  "brightness": float,          # optional
  "contrast": float             # optional
}
```

### ModelThresholds
```python
{
  "detection_threshold": float,    # 0.0 - 1.0 (default: 0.5)
  "recognition_threshold": float,  # 0.0 - 1.0 (default: 0.5)
  "min_text_size": int,            # optional
  "max_text_size": int             # optional
}
```

---

## Permission Matrix

| Action | Operator | Supervisor | Admin |
|--------|----------|------------|-------|
| List Recipes | ✅ | ✅ | ✅ |
| Load Recipe | ✅ | ✅ | ✅ |
| Search Recipes | ✅ | ✅ | ✅ |
| Create Recipe | ❌ | ✅ | ✅ |
| Update Recipe | ❌ | ✅ | ✅ |
| Delete Recipe | ❌ | ❌ | ✅ |

---

## Error Responses

### 401 Unauthorized
```json
{
  "detail": "Could not validate credentials"
}
```

### 403 Forbidden
```json
{
  "detail": "Operation not permitted. Required roles: ['supervisor', 'admin']"
}
```

### 404 Not Found
```json
{
  "detail": "Recipe not found"
}
```

### 400 Bad Request
```json
{
  "detail": "Recipe with name 'Product A' already exists"
}
```

---

## Usage Examples

### Example 1: Operator Loads Recipe for Verification

```bash
# Operator logs in
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "operator1", "password": "password123"}'

# Load recipe to verify datecode
curl -X GET http://localhost:8000/api/recipes/507f1f77bcf86cd799439011 \
  -H "Authorization: Bearer {token}"
```

### Example 2: Supervisor Creates New Recipe

```bash
# Supervisor creates new receipt for machine training
curl -X POST http://localhost:8000/api/recipes/ \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New Product Recipe",
    "product_code": "PROD-X-999",
    "camera_settings": {
      "exposure_time": 45.0,
      "delay_trigger": 90.0
    },
    "model_thresholds": {
      "detection_threshold": 0.65,
      "recognition_threshold": 0.7
    }
  }'
```

### Example 3: Supervisor Updates Recipe Content

```bash
# Supervisor edits old recipe
curl -X PUT http://localhost:8000/api/recipes/507f1f77bcf86cd799439011 \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_settings": {
      "exposure_time": 55.0,
      "delay_trigger": 110.0
    }
  }'
```

### Example 4: Admin Deletes Recipe

```bash
# Admin removes obsolete recipe
curl -X DELETE http://localhost:8000/api/recipes/507f1f77bcf86cd799439011 \
  -H "Authorization: Bearer {token}"
```

---

## Database Schema

### Collection: `recipes`

```javascript
{
  "_id": ObjectId,
  "name": String,                    // Unique with product_code
  "product_code": String,            // Unique with name
  "description": String,
  "camera_settings": {
    "exposure_time": Number,
    "delay_trigger": Number,
    "gain": Number,
    "brightness": Number,
    "contrast": Number
  },
  "model_thresholds": {
    "detection_threshold": Number,
    "recognition_threshold": Number,
    "min_text_size": Number,
    "max_text_size": Number
  },
  "template_config": Object,
  "roi_config": Object,
  "is_active": Boolean,
  "created_by": String,              // User ID
  "updated_by": String,              // User ID
  "created_at": ISODate,
  "updated_at": ISODate
}
```

### Indexes
- `name` (ascending)
- `product_code` (ascending)
- `created_by` (ascending)
- `[name, product_code]` (compound unique)

---

## Testing

Run the sample data creation script:

```bash
cd backend
python create_sample_recipes.py
```

This will create 3 sample recipes with different configurations for testing.

---

## Notes

1. **Unique Constraints**: Recipe names and product codes must be unique
2. **Audit Trail**: All create/update operations track user and timestamp
3. **Soft Delete**: Consider implementing `is_deleted` flag instead of hard delete
4. **Validation**: All thresholds are validated (0.0 - 1.0 range)
5. **Pagination**: All list endpoints support pagination to handle large datasets

---

## Future Enhancements

- [ ] Recipe versioning and history
- [ ] Recipe templates/presets
- [ ] Batch recipe operations
- [ ] Recipe export/import (JSON, CSV)
- [ ] Recipe cloning functionality
- [ ] Advanced search filters
- [ ] Recipe usage statistics
- [ ] Automated recipe optimization based on results
