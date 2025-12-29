// API Configuration
export const API_BASE_URL = 'https://suntech-vision-api.ngrok.app';

export const API_ENDPOINTS = {
  recipes: `${API_BASE_URL}/api/recipes`,
  cameras: `${API_BASE_URL}/api/cameras`,
  auth: `${API_BASE_URL}/api/auth`,
  templates: {
    upload: `${API_BASE_URL}/api/recipes/templates/upload`,
    images: (filename: string): string => `${API_BASE_URL}/api/recipes/templates/images/${filename}`
  }
} as const;
