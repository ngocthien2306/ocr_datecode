import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import type { User } from '@/types';

interface UserContextType {
  user: User | null;
  setUser: (user: User | null) => void;
  isLoading: boolean;
  hasPermission: (permission: string) => boolean;
  canAccessPage: (page: string) => boolean;
  canPerformAction: (action: string, resource?: string) => boolean;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

interface UserProviderProps {
  children: ReactNode;
}

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load user from localStorage on mount
    const loadUser = () => {
      try {
        const userData = localStorage.getItem('user');
        if (userData) {
          const parsedUser = JSON.parse(userData);
          setUser(parsedUser);
        }
      } catch (error) {
        console.error('Error loading user from localStorage:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadUser();
  }, []);

  // Permission checking functions
  const hasPermission = (permission: string): boolean => {
    if (!user) return false;

    const rolePermissions: Record<string, string[]> = {
      admin: [
        'view_dashboard',
        'view_historical',
        'view_user_management',
        'view_camera_management',
        'view_settings',
        'create_receipt',
        'update_receipt',
        'delete_receipt',
        'load_receipt',
        'edit_template',
        'delete_template',
        'change_passwords'
      ],
      supervisor: [
        'view_dashboard',
        'view_historical',
        'create_receipt',
        'update_receipt',
        'load_receipt',
        'edit_template',
        'delete_template'
      ],
      operator: [
        'view_dashboard',
        'view_historical',
        'load_receipt',
        'edit_template'
      ],
      viewer: [
        'view_dashboard',
        'view_historical'
      ]
    };

    return rolePermissions[user.role]?.includes(permission) || false;
  };

  const canAccessPage = (page: string): boolean => {
    const pagePermissions: Record<string, string> = {
      dashboard: 'view_dashboard',
      historical: 'view_historical',
      userManagement: 'view_user_management',
      cameraManagement: 'view_camera_management',
      settings: 'view_settings'
    };

    const requiredPermission = pagePermissions[page];
    return requiredPermission ? hasPermission(requiredPermission) : true;
  };

  const canPerformAction = (action: string, resource?: string): boolean => {
    const actionPermissions: Record<string, string> = {
      create: 'create_receipt',
      update: 'update_receipt',
      delete: 'delete_receipt',
      load: 'load_receipt',
      editTemplate: 'edit_template',
      deleteTemplate: 'delete_template',
      changePasswords: 'change_passwords'
    };

    const requiredPermission = actionPermissions[action];
    return requiredPermission ? hasPermission(requiredPermission) : true;
  };

  const value: UserContextType = {
    user,
    setUser,
    isLoading,
    hasPermission,
    canAccessPage,
    canPerformAction
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
};