#!/bin/bash

# Test backend API endpoints

echo "================================"
echo "Testing Backend API Endpoints"
echo "================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8000"

# Test 1: Root endpoint
echo "1. Testing root endpoint..."
response=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/)
if [ $response -eq 200 ]; then
    echo -e "${GREEN}✓ Root endpoint working (200)${NC}"
else
    echo -e "${RED}✗ Root endpoint failed ($response)${NC}"
fi
echo ""

# Test 2: Get recipes (without auth)
echo "2. Testing recipes endpoint (without auth)..."
response=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/api/recipes/)
if [ $response -eq 401 ] || [ $response -eq 403 ]; then
    echo -e "${GREEN}✓ Recipes endpoint requires auth ($response)${NC}"
elif [ $response -eq 200 ]; then
    echo -e "${GREEN}✓ Recipes endpoint working without auth (200)${NC}"
else
    echo -e "${RED}✗ Recipes endpoint failed ($response)${NC}"
    echo "Getting error details..."
    curl -s $BASE_URL/api/recipes/ | jq '.' || curl -s $BASE_URL/api/recipes/
fi
echo ""

# Test 3: Login endpoint
echo "3. Testing login endpoint..."
response=$(curl -s -X POST "$BASE_URL/api/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123" \
  -o /tmp/login_response.json \
  -w "%{http_code}")

if [ $response -eq 200 ]; then
    echo -e "${GREEN}✓ Login successful (200)${NC}"
    TOKEN=$(cat /tmp/login_response.json | jq -r '.access_token')
    echo "Token: ${TOKEN:0:20}..."
else
    echo -e "${RED}✗ Login failed ($response)${NC}"
    cat /tmp/login_response.json
fi
echo ""

# Test 4: Get recipes with auth
if [ ! -z "$TOKEN" ]; then
    echo "4. Testing recipes endpoint with auth..."
    response=$(curl -s -o /tmp/recipes_response.json -w "%{http_code}" \
      -H "Authorization: Bearer $TOKEN" \
      "$BASE_URL/api/recipes/?skip=0&limit=10")
    
    if [ $response -eq 200 ]; then
        echo -e "${GREEN}✓ Recipes endpoint working with auth (200)${NC}"
        count=$(cat /tmp/recipes_response.json | jq '. | length' 2>/dev/null || echo "unknown")
        echo "Number of recipes: $count"
    else
        echo -e "${RED}✗ Recipes endpoint failed with auth ($response)${NC}"
        echo "Error details:"
        cat /tmp/recipes_response.json
    fi
else
    echo "4. Skipping auth test (no token)"
fi
echo ""

# Test 5: Recipe count
if [ ! -z "$TOKEN" ]; then
    echo "5. Testing recipe count endpoint..."
    response=$(curl -s -o /tmp/count_response.json -w "%{http_code}" \
      -H "Authorization: Bearer $TOKEN" \
      "$BASE_URL/api/recipes/stats/count")
    
    if [ $response -eq 200 ]; then
        echo -e "${GREEN}✓ Count endpoint working (200)${NC}"
        cat /tmp/count_response.json | jq '.'
    else
        echo -e "${RED}✗ Count endpoint failed ($response)${NC}"
        cat /tmp/count_response.json
    fi
else
    echo "5. Skipping count test (no token)"
fi
echo ""

echo "================================"
echo "Test completed"
echo "================================"
