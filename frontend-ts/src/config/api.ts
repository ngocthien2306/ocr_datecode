// API Configuration
import { API_BASE_URL } from './env';

export { API_BASE_URL };

export const API_ENDPOINTS = {
  recipes: `${API_BASE_URL}/api/recipes`,
  cameras: `${API_BASE_URL}/api/cameras`,
  auth: `${API_BASE_URL}/api/auth`,
  templates: {
    upload: `${API_BASE_URL}/api/recipes/templates/upload`,
    images: (filename: string): string => `${API_BASE_URL}/api/recipes/templates/images/${filename}`
  }
} as const;
