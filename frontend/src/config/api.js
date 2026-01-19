// API Configuration
export const API_BASE_URL = 'http://localhost:8000';
export const API_ENDPOINTS = {
  recipes: `${API_BASE_URL}/api/recipes`,
  cameras: `${API_BASE_URL}/api/cameras`,
  auth: `${API_BASE_URL}/api/auth`,
  templates: {
    upload: `${API_BASE_URL}/api/recipes/templates/upload`,
    images: (filename) => `${API_BASE_URL}/api/recipes/templates/images/${filename}`
  }
};
