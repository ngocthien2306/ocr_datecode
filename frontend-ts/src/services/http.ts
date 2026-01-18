import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import { API_BASE_URL } from '@/config/env';

// Construct full API URL with /api suffix
const API_URL = `${API_BASE_URL}/api`;

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_URL,
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

export default api;
