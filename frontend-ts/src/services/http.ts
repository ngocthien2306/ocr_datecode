import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';

// const API_BASE_URL = 'https://suntech-vision-api.ngrok.app/api';
// const API_BASE_URL = 'https://quiet-corners-lead.loca.lt/api';
const API_BASE_URL = 'http://localhost:8000/api';

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

export default api;
