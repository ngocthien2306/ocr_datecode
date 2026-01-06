import api from './http';
import type { Recipe, RecipeCreate, RecipeUpdate, CountResponse, Statistics } from '@/types';

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

  deleteRecipe: async (recipeId: string): Promise<{ message?: string }> => {
    const response = await api.delete(`/recipes/${recipeId}`);
    return response.data;
  },
};

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

  getReceiptsCount: async (isActive: boolean | null = null) => {
    return recipesAPI.getRecipeCount(isActive);
  },

  createReceipt: async (receiptData: RecipeCreate): Promise<Recipe> => {
    return recipesAPI.createRecipe(receiptData);
  },

  updateReceipt: async (receiptId: string, receiptData: RecipeUpdate): Promise<Recipe> => {
    return recipesAPI.updateRecipe(receiptId, receiptData);
  },

  deleteReceipt: async (receiptId: string): Promise<{ message?: string }> => {
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

  // Record that a receipt (recipe) was loaded. Server persists load event.
  loadReceipt: async (receiptId: string): Promise<{ id: string }> => {
    const response = await api.post<{ id: string }>(`/recipes/${receiptId}/load`);
    return response.data;
  },

  // Stop a running recipe
  stopReceipt: async (receiptId: string): Promise<{ success: boolean; message: string; recipe_id: string }> => {
    const response = await api.post<{ success: boolean; message: string; recipe_id: string }>(`/recipes/${receiptId}/stop`);
    return response.data;
  },

  // Set inference mode (ONLINE/OFFLINE)
  setInferenceMode: async (receiptId: string, enabled: boolean): Promise<{ success: boolean; mode: string; enabled: boolean }> => {
    const response = await api.post<{ success: boolean; mode: string; enabled: boolean }>(
      `/recipes/${receiptId}/set-inference-mode`,
      { enabled }
    );
    return response.data;
  },

  // Get latest loaded recipe
  getLatestLoadedRecipe: async (): Promise<any> => {
    try {
      const response = await api.get<any>('/recipes/loads/latest');
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        return null; // No recipe loaded
      }
      throw error;
    }
  },

  // Get load history for a specific receipt (recipe)
  getLoadHistory: async (receiptId: string, skip = 0, limit = 100) => {
    const response = await api.get<{ items: any[]; count: number }>(`/recipes/${receiptId}/loads`, {
      params: { skip, limit },
    });
    return response.data;
  },

  // Get global load history for all receipts (requires appropriate permissions)
  getGlobalLoadHistory: async (skip = 0, limit = 100) => {
    const response = await api.get<{ items: any[]; count: number }>(`/recipes/loads`, {
      params: { skip, limit },
    });
    return response.data;
  },
};

export default recipesAPI;
