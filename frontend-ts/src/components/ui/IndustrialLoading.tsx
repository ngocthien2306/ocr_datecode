import React from 'react';

interface IndustrialLoadingProps {
  template: string;
}

const IndustrialLoading: React.FC<IndustrialLoadingProps> = ({ template }) => {
  const getLoadingContent = () => {
    switch (template) {
      case 'spinner':
        return (
          <div className="p-8 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading...</p>
          </div>
        );

      case 'pulse':
        return (
          <div className="p-8 text-center">
            <div className="animate-pulse flex space-x-4 justify-center">
              <div className="rounded-full bg-blue-600 h-8 w-8"></div>
              <div className="rounded-full bg-blue-600 h-8 w-8"></div>
              <div className="rounded-full bg-blue-600 h-8 w-8"></div>
            </div>
            <p className="mt-2 text-gray-600">Loading...</p>
          </div>
        );

      case 'bars':
        return (
          <div className="p-8 text-center">
            <div className="flex justify-center space-x-1">
              <div className="animate-bounce bg-blue-600 h-8 w-2"></div>
              <div className="animate-bounce bg-blue-600 h-8 w-2" style={{ animationDelay: '0.1s' }}></div>
              <div className="animate-bounce bg-blue-600 h-8 w-2" style={{ animationDelay: '0.2s' }}></div>
              <div className="animate-bounce bg-blue-600 h-8 w-2" style={{ animationDelay: '0.3s' }}></div>
            </div>
            <p className="mt-2 text-gray-600">Loading...</p>
          </div>
        );

      case 'dots':
        return (
          <div className="p-8 text-center">
            <div className="flex justify-center space-x-2">
              <div className="animate-ping bg-blue-600 h-3 w-3 rounded-full"></div>
              <div className="animate-ping bg-blue-600 h-3 w-3 rounded-full" style={{ animationDelay: '0.2s' }}></div>
              <div className="animate-ping bg-blue-600 h-3 w-3 rounded-full" style={{ animationDelay: '0.4s' }}></div>
            </div>
            <p className="mt-2 text-gray-600">Loading...</p>
          </div>
        );

      case 'logs':
      default:
        return (
          <div className="p-8 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading activity logs...</p>
          </div>
        );
    }
  };

  return getLoadingContent();
};

export { IndustrialLoading };