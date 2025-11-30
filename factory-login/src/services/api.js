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
  getAllRecipes: async (skip = 0, limit = 100, activeOnly = true) => {
    const response = await api.get('/recipes/', {
      params: { skip, limit, active_only: activeOnly },
    });
    return response.data;
  },

  // Get recipe by ID
  getRecipeById: async (recipeId) => {
    const response = await api.get(`/recipes/${recipeId}`);
    return response.data;
  },

  // Get recipe by name
  getRecipeByName: async (recipeName) => {
    const response = await api.get(`/recipes/name/${recipeName}`);
    return response.data;
  },

  // Create recipe
  createRecipe: async (recipeData) => {
    const response = await api.post('/recipes/', recipeData);
    return response.data;
  },

  // Update recipe
  updateRecipe: async (recipeId, recipeData) => {
    const response = await api.put(`/recipes/${recipeId}`, recipeData);
    return response.data;
  },

  // Delete recipe
  deleteRecipe: async (recipeId, hardDelete = false) => {
    const response = await api.delete(`/recipes/${recipeId}`, {
      params: { hard_delete: hardDelete },
    });
    return response.data;
  },

  // Validate datecode
  validateDatecode: async (recipeId, datecodeInput) => {
    const response = await api.post('/recipes/validate-datecode', {
      recipe_id: recipeId,
      datecode_input: datecodeInput,
    });
    return response.data;
  },
};

// Receipts API
export const receiptsAPI = {
  // Get all receipts
  getAllReceipts: async (params = {}) => {
    const response = await api.get('/receipts/', { params });
    return response.data;
  },

  // Get receipts count
  getReceiptsCount: async (params = {}) => {
    const response = await api.get('/receipts/count', { params });
    return response.data;
  },

  // Get receipt by ID
  getReceiptById: async (receiptId) => {
    const response = await api.get(`/receipts/${receiptId}`);
    return response.data;
  },

  // Create receipt
  createReceipt: async (receiptData) => {
    const response = await api.post('/receipts/', receiptData);
    return response.data;
  },

  // Update receipt
  updateReceipt: async (receiptId, receiptData) => {
    const response = await api.put(`/receipts/${receiptId}`, receiptData);
    return response.data;
  },

  // Delete receipt
  deleteReceipt: async (receiptId) => {
    const response = await api.delete(`/receipts/${receiptId}`);
    return response.data;
  },

  // Get user stats
  getUserStats: async (userId) => {
    const response = await api.get(`/receipts/stats/user/${userId}`);
    return response.data;
  },

  // Get recipe stats
  getRecipeStats: async (recipeId) => {
    const response = await api.get(`/receipts/stats/recipe/${recipeId}`);
    return response.data;
  },
};

export default api;
