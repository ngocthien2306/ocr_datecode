/**
 * Environment Variables Configuration
 *
 * Centralized access to environment variables with type safety and validation.
 * All environment variables in Vite must be prefixed with VITE_ to be exposed to the client.
 */

/**
 * Get environment variable with fallback
 */
function getEnv(key: string, fallback: string = ''): string {
  return import.meta.env[key] || fallback;
}

/**
 * API Configuration
 */
export const API_BASE_URL = getEnv('VITE_API_BASE_URL', 'http://localhost:8000');

/**
 * Application Configuration
 */
export const APP_NAME = getEnv('VITE_APP_NAME', 'OCR Datecode System');
export const APP_VERSION = getEnv('VITE_APP_VERSION', '1.0.0');

/**
 * Development Server Configuration
 */
export const DEV_SERVER_PORT = parseInt(getEnv('VITE_DEV_SERVER_PORT', '5173'));
export const DEV_SERVER_HOST = getEnv('VITE_DEV_SERVER_HOST', 'true') === 'true';

/**
 * Environment Mode
 */
export const IS_PRODUCTION = import.meta.env.PROD;
export const IS_DEVELOPMENT = import.meta.env.DEV;
export const MODE = import.meta.env.MODE;

/**
 * Log current configuration (only in development)
 */
if (IS_DEVELOPMENT) {
  console.log('[ENV] Configuration loaded:', {
    API_BASE_URL,
    APP_NAME,
    APP_VERSION,
    MODE,
  });
}

export default {
  API_BASE_URL,
  APP_NAME,
  APP_VERSION,
  DEV_SERVER_PORT,
  DEV_SERVER_HOST,
  IS_PRODUCTION,
  IS_DEVELOPMENT,
  MODE,
};
