import axios from 'axios';

const API_BASE_URL = 'https://suntech-vision-api.ngrok.app/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
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

// Auth API
export const authAPI = {
  login: async (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await api.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  },
};

// Users API
export const usersAPI = {
  // Get all users
  getAllUsers: async (skip = 0, limit = 100) => {
    const response = await api.get('/users/', { params: { skip, limit } });
    return response.data;
  },

  // Get current user
  getCurrentUser: async () => {
    const response = await api.get('/users/me');
    return response.data;
  },

  // Get user by ID
  getUserById: async (userId) => {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  },

  // Create user
  createUser: async (userData) => {
    const response = await api.post('/users/', userData);
    return response.data;
  },

  // Update user
  updateUser: async (userId, userData) => {
    const response = await api.put(`/users/${userId}`, userData);
    return response.data;
  },

  // Delete user
  deleteUser: async (userId) => {
    const response = await api.delete(`/users/${userId}`);
    return response.data;
  },

  // Change password
  changePassword: async (oldPassword, newPassword) => {
    const response = await api.post('/users/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
    return response.data;
  },

  // Reset password (admin only)
  resetPassword: async (userId, newPassword) => {
    const response = await api.post(`/users/${userId}/reset-password`, null, {
      params: { new_password: newPassword },
    });
    return response.data;
  },
};

// Recipes API
export const recipesAPI = {
  // Get all recipes
  getAllRecipes: async (skip = 0, limit = 100, isActive = null) => {
    const params = { skip, limit };
    if (isActive !== null) {
      params.is_active = isActive;
    }
    const response = await api.get('/recipes/', { params });
    return response.data;
  },

  // Search recipes
  searchRecipes: async (query, skip = 0, limit = 100) => {
    const response = await api.get('/recipes/search', {
      params: { q: query, skip, limit },
    });
    return response.data;
  },

  // Get recipe by ID
  getRecipeById: async (recipeId) => {
    const response = await api.get(`/recipes/${recipeId}`);
    return response.data;
  },

  // Get recipe count
  getRecipeCount: async (isActive = null) => {
    const params = {};
    if (isActive !== null) {
      params.is_active = isActive;
    }
    const response = await api.get('/recipes/stats/count', { params });
    return response.data;
  },

  // Create recipe (Supervisor+)
  createRecipe: async (recipeData) => {
    const response = await api.post('/recipes/', recipeData);
    return response.data;
  },

  // Update recipe (Supervisor+)
  updateRecipe: async (recipeId, recipeData) => {
    const response = await api.put(`/recipes/${recipeId}`, recipeData);
    return response.data;
  },

  // Delete recipe (Admin only)
  deleteRecipe: async (recipeId) => {
    const response = await api.delete(`/recipes/${recipeId}`);
    return response.data;
  },
};

// Production Receipts API (for UI mapping)
export const receiptsAPI = {
  // Note: Backend uses "recipes" but UI calls them "receipts"
  // This is an alias for better UI mapping
  
  // Get all receipts (maps to recipes)
  getAllReceipts: async (skip = 0, limit = 100, isActive = null) => {
    return recipesAPI.getAllRecipes(skip, limit, isActive);
  },

  // Search receipts
  searchReceipts: async (query, skip = 0, limit = 100) => {
    return recipesAPI.searchRecipes(query, skip, limit);
  },

  // Get receipt by ID
  getReceiptById: async (receiptId) => {
    return recipesAPI.getRecipeById(receiptId);
  },

  // Get receipts count
  getReceiptsCount: async (isActive = null) => {
    return recipesAPI.getRecipeCount(isActive);
  },

  // Create receipt (Supervisor+)
  createReceipt: async (receiptData) => {
    return recipesAPI.createRecipe(receiptData);
  },

  // Update receipt (Supervisor+)
  updateReceipt: async (receiptId, receiptData) => {
    return recipesAPI.updateRecipe(receiptId, receiptData);
  },

  // Delete receipt (Admin only)
  deleteReceipt: async (receiptId) => {
    return recipesAPI.deleteRecipe(receiptId);
  },

  // Get statistics for dashboard
  getStatistics: async () => {
    try {
      const [recipes, count] = await Promise.all([
        recipesAPI.getAllRecipes(0, 100, true),
        recipesAPI.getRecipeCount(true)
      ]);
      
      return {
        totalReceipts: count.count || recipes.length,
        totalProducts: recipes.length, // Can be customized based on actual data
        successRate: 98.2, // This should come from actual OCR processing stats
        recipes: recipes
      };
    } catch (error) {
      console.error('Error fetching statistics:', error);
      throw error;
    }
  },
};

export default api;
