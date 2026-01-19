import React, { useState, useEffect } from 'react';
import IndustrialTemplate from './IndustrialTemplate';
import FuturisticTemplate from './FuturisticTemplate';
import { authAPI } from '../../services/api';

interface LoginProps {
  onLoginSuccess: () => void;
  isLightMode: boolean;
}

export default function Login({ onLoginSuccess, isLightMode }: LoginProps) {
  const [loginTemplate, setLoginTemplate] = useState(() => {
    return localStorage.getItem('loginTemplate') || 'industrial';
  });
  const [username, setUsername] = useState(() => {
    return localStorage.getItem('remembered_username') || '';
  });
  const [password, setPassword] = useState(() => {
    return localStorage.getItem('remembered_password') || '';
  });
  const [rememberDevice, setRememberDevice] = useState(() => {
    return !!localStorage.getItem('remembered_username');
  });
  const [loginError, setLoginError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Listen for login template changes from Settings
    const handleLoginTemplateChange = (e: CustomEvent) => {
      setLoginTemplate(e.detail.template);
    };

    window.addEventListener('loginTemplateChanged', handleLoginTemplateChange as EventListener);

    return () => {
      window.removeEventListener('loginTemplateChanged', handleLoginTemplateChange as EventListener);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoginError('');
    setIsLoading(true);

    try {
      const response = await authAPI.login(username, password);

      // Save token and user info
      localStorage.setItem('access_token', response.access_token);
      localStorage.setItem('user', JSON.stringify(response.user));

      // Handle remember device
      if (rememberDevice) {
        localStorage.setItem('remembered_username', username);
        localStorage.setItem('remembered_password', password);
      } else {
        localStorage.removeItem('remembered_username');
        localStorage.removeItem('remembered_password');
      }

      console.log('Login successful:', response.user);
      setIsLoading(false);

      // Call success callback
      onLoginSuccess();
    } catch (error: any) {
      console.error('Login error:', error);
      const errorMessage = error.response?.data?.detail || 'Login failed. Please check your credentials.';
      setLoginError(errorMessage);
      setIsLoading(false);
    }
  };

  const handleRememberDeviceChange = (checked: boolean) => {
    setRememberDevice(checked);

    // If unchecking remember device, clear stored credentials
    if (!checked) {
      localStorage.removeItem('remembered_username');
      localStorage.removeItem('remembered_password');
    }
  };

  const templateProps = {
    username,
    password,
    rememberDevice,
    loginError,
    isLoading,
    isLightMode,
    onUsernameChange: setUsername,
    onPasswordChange: setPassword,
    onRememberDeviceChange: handleRememberDeviceChange,
    onSubmit: handleSubmit,
  };

  // Render selected template
  switch (loginTemplate) {
    case 'futuristic':
      return <FuturisticTemplate {...templateProps} />;
    case 'industrial':
    default:
      return <IndustrialTemplate {...templateProps} />;
  }
}
