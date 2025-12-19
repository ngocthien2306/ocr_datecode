import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import type {
  User,
  UserCreate,
  UserUpdate,
  Camera,
  CameraCreate,
  CameraUpdate,
  Recipe,
  RecipeCreate,
  RecipeUpdate,
  LoginResponse,
  CountResponse,
  MessageResponse,
  Statistics
} from '@/types';

const API_BASE_URL = 'https://suntech-vision-api.ngrok.app/api';
// const API_BASE_URL = 'https://quiet-corners-lead.loca.lt/api';
// const API_BASE_URL = 'http://localhost:8000/api';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Bypass-Tunnel-Reminder': 'true',
  },
});

// Add auth token + bypass header to every request
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Always bypass LocalTunnel redirect
    config.headers['Bypass-Tunnel-Reminder'] = 'true';

    return config;
  },
  (error) => Promise.reject(error)
);

// Handle response errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);

// ==================== Auth API ====================
export const authAPI = {
  login: async (username: string, password: string): Promise<LoginResponse> => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await api.post<LoginResponse>('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Bypass-Tunnel-Reminder': 'true',
      },
    });
    return response.data;
  },
};

// ==================== Users API ====================
export const usersAPI = {
  getAllUsers: async (skip = 0, limit = 100): Promise<User[]> => {
    const response = await api.get<User[]>('/users/', { params: { skip, limit } });
    return response.data;
  },

  getCurrentUser: async (): Promise<User> => {
    const response = await api.get<User>('/users/me');
    return response.data;
  },

  getUserById: async (userId: string): Promise<User> => {
    const response = await api.get<User>(`/users/${userId}`);
    return response.data;
  },

  createUser: async (userData: UserCreate): Promise<User> => {
    const response = await api.post<User>('/users/', userData);
    return response.data;
  },

  updateUser: async (userId: string, userData: UserUpdate): Promise<User> => {
    const response = await api.put<User>(`/users/${userId}`, userData);
    return response.data;
  },

  deleteUser: async (userId: string): Promise<MessageResponse> => {
    const response = await api.delete<MessageResponse>(`/users/${userId}`);
    return response.data;
  },

  changePassword: async (oldPassword: string, newPassword: string): Promise<MessageResponse> => {
    const response = await api.post<MessageResponse>('/users/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
    return response.data;
  },

  resetPassword: async (userId: string, newPassword: string): Promise<MessageResponse> => {
    const response = await api.post<MessageResponse>(`/users/${userId}/reset-password`, null, {
      params: { new_password: newPassword },
    });
    return response.data;
  },
};

// ==================== Recipes API ====================
export const recipesAPI = {
  getAllRecipes: async (skip = 0, limit = 100, isActive: boolean | null = null): Promise<Recipe[]> => {
    const params: Record<string, number | boolean> = { skip, limit };
    if (isActive !== null) {
      params.is_active = isActive;
    }
    const response = await api.get<Recipe[]>('/recipes/', { params });
    return response.data;
  },

  searchRecipes: async (query: string, skip = 0, limit = 100): Promise<Recipe[]> => {
    const response = await api.get<Recipe[]>('/recipes/search', {
      params: { q: query, skip, limit },
    });
    return response.data;
  },

  getRecipeById: async (recipeId: string): Promise<Recipe> => {
    const response = await api.get<Recipe>(`/recipes/${recipeId}`);
    return response.data;
  },

  getRecipeCount: async (isActive: boolean | null = null): Promise<CountResponse> => {
    const params: Record<string, boolean> = {};
    if (isActive !== null) {
      params.is_active = isActive;
    }
    const response = await api.get<CountResponse>('/recipes/stats/count', { params });
    return response.data;
  },

  createRecipe: async (recipeData: RecipeCreate): Promise<Recipe> => {
    const response = await api.post<Recipe>('/recipes/', recipeData);
    return response.data;
  },

  updateRecipe: async (recipeId: string, recipeData: RecipeUpdate): Promise<Recipe> => {
    const response = await api.put<Recipe>(`/recipes/${recipeId}`, recipeData);
    return response.data;
  },

  deleteRecipe: async (recipeId: string): Promise<MessageResponse> => {
    const response = await api.delete<MessageResponse>(`/recipes/${recipeId}`);
    return response.data;
  },
};

// ==================== Receipts API (Alias for Recipes) ====================
export const receiptsAPI = {
  getAllReceipts: async (skip = 0, limit = 100, isActive: boolean | null = null): Promise<Recipe[]> => {
    return recipesAPI.getAllRecipes(skip, limit, isActive);
  },

  searchReceipts: async (query: string, skip = 0, limit = 100): Promise<Recipe[]> => {
    return recipesAPI.searchRecipes(query, skip, limit);
  },

  getReceiptById: async (receiptId: string): Promise<Recipe> => {
    return recipesAPI.getRecipeById(receiptId);
  },

  getReceiptsCount: async (isActive: boolean | null = null): Promise<CountResponse> => {
    return recipesAPI.getRecipeCount(isActive);
  },

  createReceipt: async (receiptData: RecipeCreate): Promise<Recipe> => {
    return recipesAPI.createRecipe(receiptData);
  },

  updateReceipt: async (receiptId: string, receiptData: RecipeUpdate): Promise<Recipe> => {
    return recipesAPI.updateRecipe(receiptId, receiptData);
  },

  deleteReceipt: async (receiptId: string): Promise<MessageResponse> => {
    return recipesAPI.deleteRecipe(receiptId);
  },

  getStatistics: async (): Promise<Statistics & { recipes: Recipe[] }> => {
    try {
      const [recipes, count] = await Promise.all([
        recipesAPI.getAllRecipes(0, 100, true),
        recipesAPI.getRecipeCount(true),
      ]);

      return {
        totalReceipts: count.count || recipes.length,
        totalProducts: recipes.length,
        successRate: 98.2, // placeholder
        recipes: recipes,
      };
    } catch (error) {
      console.error('Error fetching statistics:', error);
      throw error;
    }
  },
};

// ==================== Cameras API ====================
export const camerasAPI = {
  getAllCameras: async (skip = 0, limit = 100): Promise<Camera[]> => {
    const response = await api.get<Camera[]>(`/cameras?skip=${skip}&limit=${limit}`);
    return response.data;
  },

  getConnectedCameras: async (): Promise<Camera[]> => {
    const response = await api.get<Camera[]>('/cameras/connected');
    return response.data;
  },

  getCameraById: async (cameraId: string): Promise<Camera> => {
    const response = await api.get<Camera>(`/cameras/${cameraId}`);
    return response.data;
  },

  getCamerasCount: async (): Promise<CountResponse> => {
    const response = await api.get<CountResponse>('/cameras/count/total');
    return response.data;
  },

  createCamera: async (cameraData: CameraCreate): Promise<Camera> => {
    const response = await api.post<Camera>('/cameras', cameraData);
    return response.data;
  },

  updateCamera: async (cameraId: string, cameraData: CameraUpdate): Promise<Camera> => {
    const response = await api.put<Camera>(`/cameras/${cameraId}`, cameraData);
    return response.data;
  },

  updateCameraConnection: async (cameraId: string, isConnected: boolean): Promise<Camera> => {
    const response = await api.patch<Camera>(`/cameras/${cameraId}/connection`, { 
      is_connected: isConnected 
    });
    return response.data;
  },

  deleteCamera: async (cameraId: string): Promise<MessageResponse> => {
    const response = await api.delete<MessageResponse>(`/cameras/${cameraId}`);
    return response.data;
  },
};

export default api;
