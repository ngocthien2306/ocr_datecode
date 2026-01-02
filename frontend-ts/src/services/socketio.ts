/**
 * SocketIO Client Service for Real-time Updates
 *
 * Installation required:
 * npm install socket.io-client
 */

import { io, Socket } from 'socket.io-client';
import { API_BASE_URL } from '@/config/api';

class SocketIOService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  /**
   * Initialize and connect to SocketIO server
   */
  connect(): void {
    if (this.socket?.connected) {
      console.log('[SocketIO] Already connected');
      return;
    }

    const token = localStorage.getItem('access_token');

    this.socket = io(API_BASE_URL, {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      auth: {
        token: token
      },
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: this.maxReconnectAttempts
    });

    this.setupEventListeners();
  }

  /**
   * Setup default event listeners
   */
  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('[SocketIO] Connected:', this.socket?.id);
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('[SocketIO] Disconnected:', reason);
    });

    this.socket.on('connect_error', (error) => {
      console.error('[SocketIO] Connection error:', error);
      this.reconnectAttempts++;

      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('[SocketIO] Max reconnection attempts reached');
        this.disconnect();
      }
    });

    this.socket.on('connection_response', (data) => {
      console.log('[SocketIO] Connection response:', data);
    });

    this.socket.on('subscribed', (data) => {
      console.log('[SocketIO] Subscribed to:', data.channel);
    });
  }

  /**
   * Subscribe to inference results channel
   */
  subscribeToInferenceResults(callback: (data: any) => void): void {
    if (!this.socket) {
      console.error('[SocketIO] Socket not connected');
      return;
    }

    // Subscribe to channel
    this.socket.emit('subscribe_inference_results');

    // Listen for new results
    this.socket.on('new_inference_result', callback);

    console.log('[SocketIO] Subscribed to inference results');
  }

  /**
   * Unsubscribe from inference results channel
   */
  unsubscribeFromInferenceResults(callback?: (data: any) => void): void {
    if (!this.socket) return;

    this.socket.emit('unsubscribe_inference_results');

    if (callback) {
      this.socket.off('new_inference_result', callback);
    } else {
      this.socket.off('new_inference_result');
    }

    console.log('[SocketIO] Unsubscribed from inference results');
  }

  /**
   * Subscribe to recipe status changes
   */
  subscribeToRecipeStatus(callback: (data: any) => void): void {
    if (!this.socket) {
      console.error('[SocketIO] Socket not connected');
      return;
    }

    // Listen for recipe status changes
    this.socket.on('recipe_status_change', callback);

    console.log('[SocketIO] Subscribed to recipe status changes');
  }

  /**
   * Unsubscribe from recipe status changes
   */
  unsubscribeFromRecipeStatus(callback?: (data: any) => void): void {
    if (!this.socket) return;

    if (callback) {
      this.socket.off('recipe_status_change', callback);
    } else {
      this.socket.off('recipe_status_change');
    }

    console.log('[SocketIO] Unsubscribed from recipe status changes');
  }

  /**
   * Disconnect from SocketIO server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      console.log('[SocketIO] Disconnected');
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }

  /**
   * Get socket instance
   */
  getSocket(): Socket | null {
    return this.socket;
  }
}

// Export singleton instance
export const socketService = new SocketIOService();
export default socketService;
