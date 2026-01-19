import api from './http';

export const uploadAPI = {
  /**
   * Upload avatar image
   * @param file - Image file to upload
   * @returns Promise with the uploaded image URL
   */
  uploadAvatar: async (file: File): Promise<string> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/upload/avatar', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data.url || response.data.file_url || response.data.avatar_url;
  },

  /**
   * Delete uploaded avatar
   * @param url - URL of the image to delete
   */
  deleteAvatar: async (url: string): Promise<void> => {
    await api.delete('/upload/avatar', {
      data: { url },
    });
  },
};
