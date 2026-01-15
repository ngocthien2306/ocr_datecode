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

    console.log('[SocketIO] Connecting to:', API_BASE_URL);
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
   * Start camera stream
   */
  startCameraStream(serialNumber: string, frameRate: number = 10, saveEnabled: boolean = false): void {
    if (!this.socket) {
      console.error('[SocketIO] Socket not connected');
      return;
    }

    console.log(`[SocketIO] Emitting start_camera_stream for ${serialNumber} at ${frameRate} FPS (save_enabled=${saveEnabled})`);
    this.socket.emit('start_camera_stream', {
      serial_number: serialNumber,
      frame_rate: frameRate,
      save_enabled: saveEnabled
    });

    console.log(`[SocketIO] Emitted start_camera_stream command`);
  }

  /**
   * Stop camera stream
   */
  stopCameraStream(serialNumber: string): void {
    if (!this.socket) {
      console.error('[SocketIO] Socket not connected');
      return;
    }

    console.log(`[SocketIO] Emitting stop_camera_stream for ${serialNumber}`);
    this.socket.emit('stop_camera_stream', {
      serial_number: serialNumber
    });

    console.log(`[SocketIO] Emitted stop_camera_stream command`);
  }

  /**
   * Subscribe to camera frames
   */
  subscribeToCameraFrames(callback: (data: any) => void): void {
    if (!this.socket) {
      console.error('[SocketIO] Socket not connected');
      return;
    }

    console.log('[SocketIO] Subscribing to camera_frame events');
    this.socket.on('camera_frame', callback);
    console.log('[SocketIO] Successfully subscribed to camera_frame events');
  }

  /**
   * Unsubscribe from camera frames
   */
  unsubscribeFromCameraFrames(callback?: (data: any) => void): void {
    if (!this.socket) return;

    if (callback) {
      this.socket.off('camera_frame', callback);
    } else {
      this.socket.off('camera_frame');
    }

    console.log('[SocketIO] Unsubscribed from camera frames');
  }

  /**
   * Listen for camera stream events
   */
  onCameraStreamStarted(callback: (data: any) => void): void {
    if (!this.socket) return;
    this.socket.on('camera_stream_started', callback);
  }

  onCameraStreamStopped(callback: (data: any) => void): void {
    if (!this.socket) return;
    this.socket.on('camera_stream_stopped', callback);
  }

  onCameraStreamError(callback: (data: any) => void): void {
    if (!this.socket) return;
    this.socket.on('camera_stream_error', callback);
  }

  /**
   * Listen for camera service status changes
   */
  onCameraServiceStatus(callback: (data: any) => void): void {
    if (!this.socket) return;
    this.socket.on('camera_service_status', callback);
    console.log('[SocketIO] Subscribed to camera_service_status events');
  }

  offCameraServiceStatus(callback?: (data: any) => void): void {
    if (!this.socket) return;

    if (callback) {
      this.socket.off('camera_service_status', callback);
    } else {
      this.socket.off('camera_service_status');
    }

    console.log('[SocketIO] Unsubscribed from camera_service_status events');
  }

  /**
   * Listen for individual camera status updates
   */
  onCameraStatusUpdate(callback: (data: any) => void): void {
    if (!this.socket) return;
    this.socket.on('camera_status_update', callback);
    console.log('[SocketIO] Subscribed to camera_status_update events');
  }

  offCameraStatusUpdate(callback?: (data: any) => void): void {
    if (!this.socket) return;

    if (callback) {
      this.socket.off('camera_status_update', callback);
    } else {
      this.socket.off('camera_status_update');
    }

    console.log('[SocketIO] Unsubscribed from camera_status_update events');
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
   * Request I/O status (start tracking)
   */
  requestIOStatus(): void {
    if (!this.socket) {
      console.error('[SocketIO] Socket not connected');
      return;
    }

    console.log('[SocketIO] Requesting I/O status...');
    this.socket.emit('request_io_status');
  }

  /**
   * Stop I/O tracking
   */
  stopIOTracking(): void {
    if (!this.socket) return;

    console.log('[SocketIO] Stopping I/O tracking...');
    this.socket.emit('stop_io_tracking');
  }

  /**
   * Listen for I/O status updates
   */
  onIOStatusUpdate(callback: (data: any) => void): void {
    if (!this.socket) return;
    this.socket.on('io_status_update', callback);
    console.log('[SocketIO] Subscribed to io_status_update events');
  }

  offIOStatusUpdate(callback?: (data: any) => void): void {
    if (!this.socket) return;

    if (callback) {
      this.socket.off('io_status_update', callback);
    } else {
      this.socket.off('io_status_update');
    }

    console.log('[SocketIO] Unsubscribed from io_status_update events');
  }

  /**
   * Set Digital Output pin value
   */
  setDOPin(pinNumber: number, value: number): void {
    if (!this.socket) {
      console.error('[SocketIO] Socket not connected');
      return;
    }

    console.log(`[SocketIO] Setting DO${pinNumber} to ${value}`);
    this.socket.emit('set_do_pin', {
      pin_number: pinNumber,
      value: value
    });
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
