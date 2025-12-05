import axios from 'axios';

// const API_BASE_URL = 'https://suntech-vision-api.ngrok.app/api';
const API_BASE_URL = 'https://quiet-corners-lead.loca.lt/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Bypass-Tunnel-Reminder': 'true',   // 👈 Bypass LocalTunnel confirm
  },
});

// Add auth token + bypass header to every request
api.interceptors.request.use(
  (config) => {
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

// Auth API
export const authAPI = {
  login: async (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await api.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Bypass-Tunnel-Reminder': 'true',
      },
    });
    return response.data;
  },
};

// Users API
export const usersAPI = {
  getAllUsers: async (skip = 0, limit = 100) => {
    const response = await api.get('/users/', { params: { skip, limit } });
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await api.get('/users/me');
    return response.data;
  },

  getUserById: async (userId) => {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  },

  createUser: async (userData) => {
    const response = await api.post('/users/', userData);
    return response.data;
  },

  updateUser: async (userId, userData) => {
    const response = await api.put(`/users/${userId}`, userData);
    return response.data;
  },

  deleteUser: async (userId) => {
    const response = await api.delete(`/users/${userId}`);
    return response.data;
  },

  changePassword: async (oldPassword, newPassword) => {
    const response = await api.post('/users/change-password', {
      old_password: oldPassword,
      new_password: newPassword,
    });
    return response.data;
  },

  resetPassword: async (userId, newPassword) => {
    const response = await api.post(`/users/${userId}/reset-password`, null, {
      params: { new_password: newPassword },
    });
    return response.data;
  },
};

// Recipes API
export const recipesAPI = {
  getAllRecipes: async (skip = 0, limit = 100, isActive = null) => {
    const params = { skip, limit };
    if (isActive !== null) {
      params.is_active = isActive;
    }
    const response = await api.get('/recipes/', { params });
    return response.data;
  },

  searchRecipes: async (query, skip = 0, limit = 100) => {
    const response = await api.get('/recipes/search', {
      params: { q: query, skip, limit },
    });
    return response.data;
  },

  getRecipeById: async (recipeId) => {
    const response = await api.get(`/recipes/${recipeId}`);
    return response.data;
  },

  getRecipeCount: async (isActive = null) => {
    const params = {};
    if (isActive !== null) {
      params.is_active = isActive;
    }
    const response = await api.get('/recipes/stats/count', { params });
    return response.data;
  },

  createRecipe: async (recipeData) => {
    const response = await api.post('/recipes/', recipeData);
    return response.data;
  },

  updateRecipe: async (recipeId, recipeData) => {
    const response = await api.put(`/recipes/${recipeId}`, recipeData);
    return response.data;
  },

  deleteRecipe: async (recipeId) => {
    const response = await api.delete(`/recipes/${recipeId}`);
    return response.data;
  },
};

// Receipts API
export const receiptsAPI = {
  getAllReceipts: async (skip = 0, limit = 100, isActive = null) => {
    return recipesAPI.getAllRecipes(skip, limit, isActive);
  },

  searchReceipts: async (query, skip = 0, limit = 100) => {
    return recipesAPI.searchRecipes(query, skip, limit);
  },

  getReceiptById: async (receiptId) => {
    return recipesAPI.getRecipeById(receiptId);
  },

  getReceiptsCount: async (isActive = null) => {
    return recipesAPI.getRecipeCount(isActive);
  },

  createReceipt: async (receiptData) => {
    return recipesAPI.createRecipe(receiptData);
  },

  updateReceipt: async (receiptId, receiptData) => {
    return recipesAPI.updateRecipe(receiptId, receiptData);
  },

  deleteReceipt: async (receiptId) => {
    return recipesAPI.deleteRecipe(receiptId);
  },

  getStatistics: async () => {
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

export default api;