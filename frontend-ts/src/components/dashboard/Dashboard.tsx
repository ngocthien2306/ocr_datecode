import { useState, useEffect, useRef } from 'react';
import '@/styles/Dashboard.css';
import UserManagement from './UserManagement';
import Receipts from '../recipe/Receipts';
import InferenceRealtime from '../inference/InferenceRealtime';
import Historical from './Historical';
import Settings from './Settings';
import Logs from './Logs';
import Documentation from './Documentation';
import System from '../../pages/System';
import CameraManagement from '../camera/CameraManagement';
import ConfirmDialog from '../shared/ConfirmDialog';
import RecipeLoadingTemplates from '../shared/RecipeLoadingTemplates';
import RecipeChart from './RecipeChart';
import { camerasAPI } from '@/services/api';
import { actionLogsAPI, usersAPI } from '@/services/api';
import { receiptsAPI } from '@/services/recipes';
import { inferenceResultsAPI } from '@/services/inferenceResults';
import { socketService } from '@/services/socketio';
import { API_BASE_URL } from '@/config/api';
import { useUser } from '@/contexts/UserContext';
import type { Camera as BaseCamera, ReceiptLoad } from '@/types';
import { AgentChatWidget } from '../agent';

interface DashboardCamera extends Omit<BaseCamera, 'status'> {
  is_connected: boolean;
  location?: string;
  model_name: string;
  serial_number?: string;
  max_frame_rate?: number;
  resolution_width: number;
  resolution_height: number;
  is_active: boolean;
}

interface DashboardProps {
  onLogout: () => void;
}

type Section = 'dashboard' | 'users' | 'receipts' | 'realtime' | 'cameras' | 'historical' | 'settings' | 'logs' | 'documentation' | 'system';

type LoadingTemplate = 'camera-vision' | 'users' | 'receipts' | 'cameras' | 'historical' | 'settings' | 'logs' | 'documentation' | 'spinner' | 'pulse' | 'radar' | 'grid' | 'circuit' | 'barcode' | 'ocr' | 'dots' | 'waves' | 'ocr-scanner' | 'barcode-scanner' | 'neural-network' | 'qr-detector' | 'industrial-factory' | 'ai-vision-pipeline' | 'neural-processing' | 'system-io';

interface LoadingTemplates {
  dashboard: LoadingTemplate;
  users: LoadingTemplate;
  receipts: LoadingTemplate;
  cameras: LoadingTemplate;
  historical: LoadingTemplate;
  settings: LoadingTemplate;
  logs: LoadingTemplate;
  documentation: LoadingTemplate;
  realtime: LoadingTemplate;
  system: LoadingTemplate;
}

interface LoadingBackground {
  image: string;
  opacity: number;
}

interface ConfirmDialogState {
  isOpen: boolean;
  title: string;
  message: string;
  type: 'warning' | 'danger' | 'info';
  onConfirm: (() => void) | null;
}

interface CameraStats {
  total: number;
  connected: number;
  active: number;
}

export default function Dashboard({ onLogout }: DashboardProps) {
  const { canAccessPage } = useUser();
  const [currentSection, setCurrentSection] = useState<Section>('dashboard');
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingTemplates, setLoadingTemplates] = useState<LoadingTemplates>(() => {
    return {
      dashboard: (localStorage.getItem('dashboardLoading') as LoadingTemplate) || 'camera-vision',
      users: (localStorage.getItem('usersLoading') as LoadingTemplate) || 'users',
      receipts: (localStorage.getItem('receiptsLoading') as LoadingTemplate) || 'receipts',
      cameras: (localStorage.getItem('camerasLoading') as LoadingTemplate) || 'cameras',
      historical: (localStorage.getItem('historicalLoading') as LoadingTemplate) || 'historical',
      settings: (localStorage.getItem('settingsLoading') as LoadingTemplate) || 'settings',
      logs: (localStorage.getItem('logsLoading') as LoadingTemplate) || 'logs',
      documentation: (localStorage.getItem('documentationLoading') as LoadingTemplate) || 'ocr',
      realtime: (localStorage.getItem('realtimeLoading') as LoadingTemplate) || 'realtime',
      system: (localStorage.getItem('systemLoading') as LoadingTemplate) || 'system-io'
    };
  });
  const [loadingBackground, setLoadingBackground] = useState<LoadingBackground>(() => {
    return {
      image: localStorage.getItem('loadingBackground') || 'background1',
      opacity: parseFloat(localStorage.getItem('loadingBackgroundOpacity') || '0.2')
    };
  });
  const [pageLoadingOverlayMode, setPageLoadingOverlayMode] = useState<'fullscreen' | 'dashboard-main'>(() => {
    return (localStorage.getItem('pageLoadingOverlayMode') as 'fullscreen' | 'dashboard-main') || 'fullscreen';
  });
  const [sidebarPosition, setSidebarPosition] = useState<'left' | 'right' | 'top' | 'bottom'>(() => {
    return (localStorage.getItem('sidebarPosition') as 'left' | 'right' | 'top' | 'bottom') || 'left';
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    return localStorage.getItem('sidebarCollapsed') === 'true';
  });
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('appThemeMode');
    const isDark = savedMode === 'dark';
    // Set initial dark mode class on mount
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    return isDark;
  });
  const [currentTime, setCurrentTime] = useState(new Date());
  const [totalProductsToday, setTotalProductsToday] = useState(0);
  const [todayStats, setTodayStats] = useState({
    total: 0,
    pass: 0,
    fail: 0,
    passRate: 0,
    yesterdayTotal: 0,
    changePercent: 0
  });
  const [cameras, setCameras] = useState<DashboardCamera[]>([]);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    title: '',
    message: '',
    type: 'warning',
    onConfirm: null
  });
  const [cameraStats, setCameraStats] = useState<CameraStats>({
    total: 0,
    connected: 0,
    active: 0
  });
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [recentLoads, setRecentLoads] = useState<ReceiptLoad[]>([]);
  const [recentMembers, setRecentMembers] = useState<{username: string; full_name?: string; last_seen?: string; avatar_url?: string | null}[]>([]);
  const [runningRecipeId, setRunningRecipeId] = useState<string | null>(null);

  // Recipe charts data
  interface RecipeChartData {
    recipeId: string;
    recipeName: string;
    oneHour: { labels: string[]; totalData: number[]; passData: number[]; failData: number[] };
    today: { labels: string[]; totalData: number[]; passData: number[]; failData: number[] };
    month: { labels: string[]; totalData: number[]; passData: number[]; failData: number[] };
  }
  const [recipeCharts, setRecipeCharts] = useState<RecipeChartData[]>([]);
  const [activeRecipes, setActiveRecipes] = useState<{id: string; name: string}[]>([]);

  // Camera frames state
  const [cameraFrames, setCameraFrames] = useState<Record<string, string>>({});

  // Helper function to render loading template
  const renderLoadingTemplate = (template: LoadingTemplate) => {
    switch(template) {
      case 'camera-vision':
        return (
          <>
            <div className="camera-focus-ring">
              <div className="scan-lines">
                <div className="scan-line"></div>
                <div className="scan-line"></div>
              </div>
              <div className="focus-ring"></div>
              <div className="focus-ring"></div>
              <div className="focus-ring"></div>
              <div className="focus-ring"></div>
              <div className="camera-lens"></div>
              <div className="corner-brackets">
                <div className="bracket top-left"></div>
                <div className="bracket top-right"></div>
                <div className="bracket bottom-left"></div>
                <div className="bracket bottom-right"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">VISION SYSTEM</div>
              <div className="loading-subtitle">INITIALIZING...</div>
            </div>
            <div className="progress-indicators">
              <div className="progress-bar-container">
                <div className="progress-bar-fill"></div>
              </div>
              <div className="status-dots">
                <div className="status-dot"></div>
                <div className="status-dot"></div>
                <div className="status-dot"></div>
                <div className="status-dot"></div>
              </div>
            </div>
            <div className="system-info">
              <div className="system-info-item">
                <span className="status-indicator"></span>
                <span>CAMERA MODULE</span>
              </div>
              <div className="system-info-item">
                <span className="status-indicator"></span>
                <span>OCR ENGINE</span>
              </div>
              <div className="system-info-item">
                <span className="status-indicator"></span>
                <span>DATABASE</span>
              </div>
            </div>
          </>
        );

      case 'spinner':
        return (
          <>
            <div className="spinner-loader">
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">LOADING</div>
              <div className="loading-subtitle">PLEASE WAIT...</div>
            </div>
          </>
        );

      case 'pulse':
        return (
          <>
            <div className="pulse-loader">
              <div className="pulse-ring"></div>
              <div className="pulse-ring"></div>
              <div className="pulse-ring"></div>
              <div className="pulse-core"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">PROCESSING</div>
              <div className="loading-subtitle">ANALYZING DATA...</div>
            </div>
          </>
        );

      case 'radar':
        return (
          <>
            <div className="radar-loader">
              <div className="radar-grid">
                <div className="radar-circle"></div>
                <div className="radar-circle"></div>
                <div className="radar-circle"></div>
                <div className="radar-circle"></div>
              </div>
              <div className="radar-sweep"></div>
              <div className="radar-dot"></div>
              <div className="radar-dot"></div>
              <div className="radar-dot"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">SCANNING</div>
              <div className="loading-subtitle">DETECTING OBJECTS...</div>
            </div>
          </>
        );

      case 'grid':
        return (
          <>
            <div className="grid-loader">
              <div className="grid-container">
                {[...Array(25)].map((_, i) => (
                  <div key={i} className="grid-cell"></div>
                ))}
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">MATRIX LOADING</div>
              <div className="loading-subtitle">SYNCING SYSTEMS...</div>
            </div>
          </>
        );

      case 'circuit':
        return (
          <>
            <div className="circuit-loader">
              <div className="circuit-board">
                <div className="circuit-line horizontal line-1"></div>
                <div className="circuit-line horizontal line-2"></div>
                <div className="circuit-line vertical line-3"></div>
                <div className="circuit-line vertical line-4"></div>
                <div className="circuit-node node-1"></div>
                <div className="circuit-node node-2"></div>
                <div className="circuit-node node-3"></div>
                <div className="circuit-node node-4"></div>
                <div className="circuit-chip"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">SYSTEM BOOT</div>
              <div className="loading-subtitle">INITIALIZING HARDWARE...</div>
            </div>
          </>
        );

      case 'barcode':
        return (
          <>
            <div className="barcode-loader">
              <div className="barcode-container">
                {[...Array(15)].map((_, i) => (
                  <div key={i} className="barcode-line"></div>
                ))}
                <div className="scanner-beam"></div>
              </div>
              <div className="scanner-frame">
                <div className="scanner-corner top-left"></div>
                <div className="scanner-corner top-right"></div>
                <div className="scanner-corner bottom-left"></div>
                <div className="scanner-corner bottom-right"></div>
              </div>
              <div className="barcode-digits">8 901234 567890</div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">BARCODE SCANNING</div>
              <div className="loading-subtitle">READING CODE...</div>
            </div>
          </>
        );

      case 'ocr':
        return (
          <>
            <div className="ocr-loader">
              <div className="ocr-document">
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-scan-overlay"></div>
                <div className="ocr-highlight-box"></div>
                <div className="ocr-highlight-box"></div>
                <div className="ocr-highlight-box"></div>
              </div>
              <div className="ocr-analysis-indicators">
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">A</div>
                  <div className="ocr-indicator-label">TEXT</div>
                </div>
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">1</div>
                  <div className="ocr-indicator-label">DIGIT</div>
                </div>
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">✓</div>
                  <div className="ocr-indicator-label">VERIFY</div>
                </div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">OCR PROCESSING</div>
              <div className="loading-subtitle">RECOGNIZING TEXT...</div>
            </div>
          </>
        );

      case 'users':
        return (
          <>
            <div className="users-loader">
              <div className="users-circle-container">
                <div className="user-avatar"></div>
                <div className="user-avatar"></div>
                <div className="user-avatar"></div>
              </div>
              <div className="users-connection"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">USER MANAGEMENT</div>
              <div className="loading-subtitle">LOADING USERS...</div>
            </div>
          </>
        );

      case 'receipts':
        return (
          <>
            <div className="receipts-loader">
              <div className="receipt-paper">
                <div className="receipt-header">
                  <div className="receipt-icon"></div>
                  <div className="receipt-title"></div>
                </div>
                <div className="receipt-line"></div>
                <div className="receipt-line"></div>
                <div className="receipt-line"></div>
                <div className="receipt-line"></div>
                <div className="receipt-scanner-line"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">RECIPES</div>
              <div className="loading-subtitle">SCANNING DOCUMENTS...</div>
            </div>
          </>
        );

      case 'cameras':
        return (
          <>
            <div className="cameras-loader">
              <div className="camera-grid">
                <div className="camera-lens"></div>
                <div className="camera-lens"></div>
                <div className="camera-lens"></div>
                <div className="camera-lens"></div>
              </div>
              <div className="camera-status-indicator">
                <div className="camera-dot"></div>
                <div className="camera-dot"></div>
                <div className="camera-dot"></div>
                <div className="camera-dot"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">CAMERAS</div>
              <div className="loading-subtitle">CONNECTING DEVICES...</div>
            </div>
          </>
        );

      case 'historical':
        return (
          <>
            <div className="historical-loader">
              <div className="timeline-container">
                <div className="timeline-line"></div>
                <div className="timeline-point"></div>
                <div className="timeline-point"></div>
                <div className="timeline-point"></div>
                <div className="timeline-card"></div>
                <div className="timeline-card"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">HISTORICAL DATA</div>
              <div className="loading-subtitle">LOADING TIMELINE...</div>
            </div>
          </>
        );

      case 'settings':
        return (
          <>
            <div className="settings-loader">
              <div className="gears-container">
                <div className="gear gear-large"></div>
                <div className="gear gear-small-1"></div>
                <div className="gear gear-small-2"></div>
                <div className="settings-particles">
                  <div className="settings-particle"></div>
                  <div className="settings-particle"></div>
                  <div className="settings-particle"></div>
                </div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">SETTINGS</div>
              <div className="loading-subtitle">CONFIGURING...</div>
            </div>
          </>
        );

      case 'logs':
        return (
          <>
            <div className="logs-loader">
              <div className="logs-container">
                <div className="log-entry"></div>
                <div className="log-entry"></div>
                <div className="log-entry"></div>
                <div className="log-entry"></div>
                <div className="log-entry"></div>
                <div className="log-scroll-indicator"></div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">LOGS</div>
              <div className="loading-subtitle">LOADING AUDIT TRAIL...</div>
            </div>
          </>
        );

      case 'documentation':
        return (
          <>
            <div className="ocr-loader">
              <div className="ocr-document">
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-text-line"></div>
                <div className="ocr-scan-overlay"></div>
                <div className="ocr-highlight-box"></div>
                <div className="ocr-highlight-box"></div>
                <div className="ocr-highlight-box"></div>
              </div>
              <div className="ocr-analysis-indicators">
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">📚</div>
                  <div className="ocr-indicator-label">DOCS</div>
                </div>
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">📖</div>
                  <div className="ocr-indicator-label">GUIDE</div>
                </div>
                <div className="ocr-indicator">
                  <div className="ocr-indicator-icon">✓</div>
                  <div className="ocr-indicator-label">HELP</div>
                </div>
              </div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">DOCUMENTATION</div>
              <div className="loading-subtitle">LOADING GUIDES...</div>
            </div>
          </>
        );

      case 'system-io':
        return (
          <>
            <div className="system-io-loader">
              <div className="io-pins-container">
                <div className="io-section">
                  <div className="io-label">DI</div>
                  <div className="io-pins">
                    <div className="io-pin io-pin-active"></div>
                    <div className="io-pin"></div>
                    <div className="io-pin io-pin-active"></div>
                    <div className="io-pin"></div>
                  </div>
                </div>
                <div className="io-section">
                  <div className="io-label">DO</div>
                  <div className="io-pins">
                    <div className="io-pin io-pin-output"></div>
                    <div className="io-pin io-pin-output io-pin-active"></div>
                    <div className="io-pin io-pin-output"></div>
                    <div className="io-pin io-pin-output io-pin-active"></div>
                  </div>
                </div>
              </div>
              <div className="io-signal-wave"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">SYSTEM I/O</div>
              <div className="loading-subtitle">INITIALIZING PINS...</div>
            </div>
          </>
        );

      case 'ocr-scanner':
      case 'barcode-scanner':
      case 'neural-network':
      case 'qr-detector':
      case 'industrial-factory':
      case 'ai-vision-pipeline':
      case 'neural-processing':
        return (
          <RecipeLoadingTemplates
            template={template}
            progress={loadingProgress}
            isLightMode={!darkMode}
            overlayMode="fullscreen"
          />
        );

      default:
        return (
          <>
            <div className="spinner-loader">
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
              <div className="spinner-circle"></div>
            </div>
            <div className="loading-text-container">
              <div className="loading-title">LOADING</div>
              <div className="loading-subtitle">PLEASE WAIT...</div>
            </div>
          </>
        );
    }
  };

  // Fetch recipe charts data
  const fetchRecipeCharts = async () => {
    try {
      // Get active recipes from today's stats
      const now = new Date();

      // Today: 00:00:00 to now (VN time)
      const startOfToday = new Date(now);
      startOfToday.setHours(0, 0, 0, 0);

      const todayData = await inferenceResultsAPI.getSummaryStatistics({
        start_date: startOfToday.toISOString(),
        end_date: now.toISOString()
      });

      // Get unique recipes from today
      const recipes = todayData.by_recipe.map(r => ({
        id: r.recipe_id,
        name: r.recipe_name
      }));
      setActiveRecipes(recipes);

      if (recipes.length === 0) {
        setRecipeCharts([]);
        return;
      }

      // Fetch charts for each recipe
      const chartsData = await Promise.all(recipes.map(async (recipe) => {
        // 1 Hour (minute data): last 1 hour VN time
        const oneHourStart = new Date(now);
        oneHourStart.setHours(now.getHours() - 1);
        const oneHourStats = await inferenceResultsAPI.getTimeseriesStatistics({
          start_date: oneHourStart.toISOString(),
          end_date: now.toISOString(),
          granularity: 'minute',
          recipe_ids: recipe.id
        });

        // Today (24 hours - hourly data): 00:00:00 to now VN time
        const todayStart = new Date(now);
        todayStart.setHours(0, 0, 0, 0);

        console.log('=== Dashboard Recipe Charts ===');
        console.log('Now:', now);
        console.log('Today Start (VN 00:00):', todayStart);
        console.log('Start ISO:', todayStart.toISOString());
        console.log('End ISO:', now.toISOString());

        const todayStats = await inferenceResultsAPI.getTimeseriesStatistics({
          start_date: todayStart.toISOString(),
          end_date: now.toISOString(),
          granularity: 'hour',
          recipe_ids: recipe.id
        });

        console.log('Response data points:', todayStats.data?.length || 0);
        if (todayStats.data && todayStats.data.length > 0) {
          console.log('First timestamp:', todayStats.data[0]!.timestamp);
          console.log('Last timestamp:', todayStats.data[todayStats.data.length - 1]!.timestamp);
        }

        // Month (30 days - daily data): last 30 days VN time
        const monthStart = new Date(now);
        monthStart.setDate(now.getDate() - 30);
        monthStart.setHours(0, 0, 0, 0);
        const monthStats = await inferenceResultsAPI.getTimeseriesStatistics({
          start_date: monthStart.toISOString(),
          end_date: now.toISOString(),
          granularity: 'day',
          recipe_ids: recipe.id
        });

        // Note: Backend already returns VN timezone, no conversion needed
        return {
          recipeId: recipe.id,
          recipeName: recipe.name,
          oneHour: {
            labels: oneHourStats.data.map(d => {
              const date = new Date(d.timestamp);
              return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            }),
            totalData: oneHourStats.data.map(d => d.total),
            passData: oneHourStats.data.map(d => d.pass),
            failData: oneHourStats.data.map(d => d.fail)
          },
          today: {
            labels: todayStats.data.map(d => {
              const date = new Date(d.timestamp);
              return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            }),
            totalData: todayStats.data.map(d => d.total),
            passData: todayStats.data.map(d => d.pass),
            failData: todayStats.data.map(d => d.fail)
          },
          month: {
            labels: monthStats.data.map(d => {
              const date = new Date(d.timestamp);
              return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }),
            totalData: monthStats.data.map(d => d.total),
            passData: monthStats.data.map(d => d.pass),
            failData: monthStats.data.map(d => d.fail)
          }
        };
      }));

      setRecipeCharts(chartsData);
    } catch (error) {
      console.error('Error fetching recipe charts:', error);
    }
  };

  // Fetch today's statistics
  const fetchTodayStatistics = async () => {
    try {
      // Use VN timezone - backend will handle UTC conversion
      const now = new Date();

      // Today: 00:00:00 to 23:59:59 VN time
      const startOfToday = new Date(now);
      startOfToday.setHours(0, 0, 0, 0);
      const endOfToday = new Date(now);
      endOfToday.setHours(23, 59, 59, 999);

      // Yesterday: 00:00:00 to 23:59:59 VN time
      const startOfYesterday = new Date(now);
      startOfYesterday.setDate(now.getDate() - 1);
      startOfYesterday.setHours(0, 0, 0, 0);
      const endOfYesterday = new Date(now);
      endOfYesterday.setDate(now.getDate() - 1);
      endOfYesterday.setHours(23, 59, 59, 999);

      // Fetch today's stats
      const todayData = await inferenceResultsAPI.getSummaryStatistics({
        start_date: startOfToday.toISOString(),
        end_date: endOfToday.toISOString()
      });

      // Fetch yesterday's stats for comparison
      const yesterdayData = await inferenceResultsAPI.getSummaryStatistics({
        start_date: startOfYesterday.toISOString(),
        end_date: endOfYesterday.toISOString()
      });

      // Calculate change percentage
      const changePercent = yesterdayData.total > 0
        ? ((todayData.total - yesterdayData.total) / yesterdayData.total) * 100
        : 0;

      setTodayStats({
        total: todayData.total,
        pass: todayData.pass,
        fail: todayData.fail,
        passRate: todayData.pass_rate,
        yesterdayTotal: yesterdayData.total,
        changePercent: Math.round(changePercent * 10) / 10 // Round to 1 decimal
      });

      setTotalProductsToday(todayData.total);
    } catch (error) {
      console.error('Error fetching today statistics:', error);
    }
  };

  // Fetch cameras on component mount
  useEffect(() => {
    fetchCameras();
    fetchRecentLoads();
    fetchRecentMembers();
    fetchTodayStatistics();
    fetchRecipeCharts();
  }, []);

  // Auto-refresh today stats and charts every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchTodayStatistics();
      fetchRecipeCharts();
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Setup socket connection for camera streams
  useEffect(() => {
    if (currentSection !== 'dashboard') return;

    // Connect socket
    socketService.connect();

    // Subscribe to camera frames
    const handleCameraFrame = (data: any) => {
      const { serial_number, frame_base64 } = data;
      if (frame_base64) {
        setCameraFrames(prev => ({
          ...prev,
          [serial_number]: `data:image/jpeg;base64,${frame_base64}`
        }));
      }
    };

    socketService.subscribeToCameraFrames(handleCameraFrame);

    // Start streams for all connected cameras with low quality (resize 5x)
    cameras.forEach(camera => {
      if (camera.is_connected) {
        const serialNumber = camera.serial_number || camera.camera_id;
        // Low FPS for dashboard preview, resize 5x for low quality
        socketService.startCameraStream(serialNumber, 15, false); // 2 FPS, no save
      }
    });

    return () => {
      // Stop all streams on unmount
      cameras.forEach(camera => {
        if (camera.is_connected) {
          const serialNumber = camera.serial_number || camera.camera_id;
          socketService.stopCameraStream(serialNumber);
        }
      });
      socketService.unsubscribeFromCameraFrames(handleCameraFrame);
    };
  }, [currentSection, cameras]);

  // Load current user info
  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (userData) {
      try {
        const user = JSON.parse(userData);
        setCurrentUser(user);
      } catch (error) {
        console.error('Error parsing user data:', error);
      }
    }
  }, []);

  // Listen for loading template changes from Settings
  useEffect(() => {
    const handleTemplateChange = (event: any) => {
      const { tab, template } = event.detail;
      const tabName = tab.replace('Loading', '') as keyof LoadingTemplates;
      setLoadingTemplates(prev => ({
        ...prev,
        [tabName]: template
      }));
    };

    const handleBackgroundChange = (event: any) => {
      const { background, opacity } = event.detail;
      setLoadingBackground({
        image: background,
        opacity: opacity
      });
    };

    const handlePageOverlayModeChange = (event: any) => {
      const { mode } = event.detail;
      setPageLoadingOverlayMode(mode);
    };

    const handleSidebarPositionChange = (event: any) => {
      const { position } = event.detail;
      setSidebarPosition(position);
    };

    window.addEventListener('tabLoadingChanged', handleTemplateChange);
    window.addEventListener('loadingBackgroundChanged', handleBackgroundChange);
    window.addEventListener('pageLoadingOverlayModeChanged', handlePageOverlayModeChange);
    window.addEventListener('sidebarPositionChanged', handleSidebarPositionChange);

    return () => {
      window.removeEventListener('tabLoadingChanged', handleTemplateChange);
      window.removeEventListener('loadingBackgroundChanged', handleBackgroundChange);
      window.removeEventListener('pageLoadingOverlayModeChanged', handlePageOverlayModeChange);
      window.removeEventListener('sidebarPositionChanged', handleSidebarPositionChange);
    };
  }, []);

  // Animate loading progress
  useEffect(() => {
    if (isLoading) {
      setLoadingProgress(0);
      const interval = setInterval(() => {
        setLoadingProgress(prev => {
          if (prev >= 95) {
            clearInterval(interval);
            return prev;
          }
          return prev + Math.random() * 10;
        });
      }, 100);

      return () => clearInterval(interval);
    } else {
      setLoadingProgress(100);
    }
  }, [isLoading]);

  // Check for running recipe (initial load + subscribe to realtime changes)
  useEffect(() => {
    const checkRunningRecipe = async () => {
      try {
        const latest = await receiptsAPI.getLatestLoadedRecipe();
        if (latest && latest.recipe_id) {
          setRunningRecipeId(latest.recipe_id);
        } else {
          setRunningRecipeId(null);
        }
      } catch (error) {
        // No running recipe (404 is expected)
        setRunningRecipeId(null);
      }
    };

    // Initial check
    checkRunningRecipe();

    // Subscribe to realtime recipe status changes via SocketIO
    import('@/services/socketio').then(({ socketService }) => {
      socketService.connect();

      const handleRecipeStatusChange = (data: any) => {
        console.log('[Dashboard] Recipe status change:', data);

        if (data.event === 'recipe_loaded') {
          // Recipe was loaded
          setRunningRecipeId(data.recipe_id);
        } else if (data.event === 'recipe_stopped') {
          // Recipe was stopped
          setRunningRecipeId(null);
        }
      };

      socketService.subscribeToRecipeStatus(handleRecipeStatusChange);

      // Cleanup on unmount
      return () => {
        socketService.unsubscribeFromRecipeStatus(handleRecipeStatusChange);
      };
    });
  }, []);

  const fetchCameras = async () => {
    const startTime = Date.now();
    
    try {
      const [allCameras, countData] = await Promise.all([
        camerasAPI.getAllCameras(0, 20),
        camerasAPI.getCamerasCount()
      ]);
      
      setCameras(allCameras as unknown as DashboardCamera[]);
      
      const connectedCount = (allCameras as unknown as DashboardCamera[]).filter(c => c.is_connected).length;
      const activeCount = (allCameras as unknown as DashboardCamera[]).filter(c => c.is_active).length;
      
      setCameraStats({
        total: countData.count || allCameras.length,
        connected: connectedCount,
        active: activeCount
      });
    } catch (err) {
      console.error('Error fetching cameras:', err);
    } finally {
      // Ensure minimum loading time of 1000ms
      const elapsedTime = Date.now() - startTime;
      const remainingTime = Math.max(0, 1000 - elapsedTime);
      
      setTimeout(() => {
        setIsLoading(false);
      }, remainingTime);
    }
  };

  const fetchRecentLoads = async () => {
    try {
      const response = await receiptsAPI.getGlobalLoadHistory(0, 10); // Get top 10 recent loads
      setRecentLoads(response.items as ReceiptLoad[]);
    } catch (err) {
      console.error('Error fetching recent loads:', err);
    }
  };

  const fetchRecentMembers = async (topK = 4) => {
    try {
      // get recent action logs (backend returns timestamps in configured TZ)
      const logs = await actionLogsAPI.getRecentActionLogs(50);
      // filter for login actions and dedupe by username preserving order
      const seen = new Set<string>();
      const members: {username: string; last_seen?: string}[] = [];
      for (const l of logs) {
        if (l.action_type !== 'login') continue;
        if (!l.username) continue;
        if (seen.has(l.username)) continue;
        seen.add(l.username);
        members.push({ username: l.username, last_seen: l.timestamp });
        if (members.length >= topK) break;
      }

      // fetch simple user list to get full names and avatars (optional)
      let fullnameMap: Record<string, string> = {};
      let avatarMap: Record<string, string | undefined> = {};
      try {
        const users = await usersAPI.getSimpleUsers(0, 200);
        users.forEach(u => {
          if (u.username) {
            fullnameMap[u.username] = u.full_name || '';
            // normalize empty string to undefined
            avatarMap[u.username] = u.avatar_url && u.avatar_url.length > 0 ? u.avatar_url : undefined;
          }
        });
      } catch (e) {
        // ignore permission errors
      }

      // format last_seen nicely
      const formatted = members.map(m => ({
        username: m.username,
        full_name: fullnameMap[m.username] || undefined,
        avatar_url: avatarMap[m.username] || undefined,
        last_seen: m.last_seen ? (new Date(m.last_seen)).toLocaleString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true }) : undefined
      }));

      setRecentMembers(formatted);
    } catch (err) {
      console.error('Error fetching recent members:', err);
    }
  };

  // Handle section change with loading
  const handleSectionChange = (section: Section) => {
    setIsLoading(true);
    setCurrentSection(section);
    
    // Simulate loading time of 1000ms
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  };

  // Dark mode effect
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode');
      document.documentElement.classList.add('dark');
      localStorage.setItem('appThemeMode', 'dark');
    } else {
      document.body.classList.remove('dark-mode');
      document.documentElement.classList.remove('dark');
      localStorage.setItem('appThemeMode', 'light');
    }
  }, [darkMode]);

  // Update time every minute
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 60000);
    return () => clearInterval(timer);
  }, []);


  const handleLogout = () => {
    setConfirmDialog({
      isOpen: true,
      title: 'Logout',
      message: 'Are you sure you want to logout?',
      type: 'warning',
      onConfirm: () => {
        document.body.classList.remove('dark-mode');
        document.documentElement.classList.remove('dark');
        onLogout();
      }
    });
  };

  const timeString = currentTime.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });

  const dateString = currentTime.toLocaleDateString('en-US', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric'
  });

  return (
    <div className="dashboard-container">
      {/* Background for entire dashboard-container */}
      {loadingBackground.image !== 'none' && (
        <div 
          className="dashboard-background"
          style={{
            backgroundImage: `url(/background/${loadingBackground.image}.png)`,
            opacity: loadingBackground.opacity
          }}
        />
      )}
      
      {/* Header */}
      <header className="dashboard-header-top">
        <div className="header-left">
          <div className="dashboard-logo-section">
            <div className="company-logo">
              <img src="logo.png" alt="Suntech Automation" className="h-16 w-auto" />
            </div>
            <div className="dashboard-logo-divider"></div>
            <div className="dashboard-company-info">
              <h1 className="dashboard-company-name">SUNTECH AUTOMATION</h1>
              <p className="dashboard-version">v1.0.0 PRODUCTION</p>
            </div>
          </div>
        </div>
        <div className="header-center">
          <div className="search-bar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="8" stroke="currentColor" strokeWidth="2"/>
              <path d="m21 21-4.35-4.35" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <input type="text" placeholder="Search cameras, receipts..." />
          </div>
        </div>
        <div className="header-right">
          <div className="connection-status">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="2" fill={cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? "#10b981" : "#f59e0b"}/>
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" stroke="currentColor" strokeWidth="2"/>
            </svg>
            <span>{cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? 'All cameras online' : `${cameraStats.connected}/${cameraStats.total} cameras online`}</span>
          </div>
          <div className="profile">
            {currentUser?.avatar_url ? (
              <img 
                src={`${API_BASE_URL}${currentUser.avatar_url}`} 
                alt="User Avatar" 
                className="profile-avatar" 
              />
            ) : (
              <div className="profile-avatar-placeholder">
                {currentUser?.full_name?.charAt(0) || currentUser?.username?.charAt(0) || 'U'}
              </div>
            )}
            <span>{currentUser?.full_name || currentUser?.username || 'User'}</span>
          </div>
        </div>
      </header>

      <div className={`main-content-wrapper sidebar-${sidebarPosition} ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        {/* Sidebar */}
        <aside className="sidebar">
          {/* Toggle button for left/right sidebar */}
          {(sidebarPosition === 'left' || sidebarPosition === 'right') && (
            <button
              className="sidebar-toggle"
              onClick={() => {
                const newState = !sidebarCollapsed;
                setSidebarCollapsed(newState);
                localStorage.setItem('sidebarCollapsed', String(newState));
              }}
              title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {sidebarCollapsed ? (
                sidebarPosition === 'left' ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )
              ) : (
                sidebarPosition === 'left' ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )
              )}
            </button>
          )}
          <h3>Menu</h3>
          <nav className="nav-menu">
            <a
              href="#dashboard"
              className={`nav-item ${currentSection === 'dashboard' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('dashboard'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="3" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
                <rect x="14" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
                <rect x="3" y="14" width="7" height="7" stroke="currentColor" strokeWidth="2"/>
              </svg>
              Dashboard
            </a>
            {canAccessPage('userManagement') && (
              <a
                href="#users"
                className={`nav-item ${currentSection === 'users' ? 'active' : ''}`}
                onClick={(e) => { e.preventDefault(); handleSectionChange('users'); }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <circle cx="9" cy="7" r="4" stroke="currentColor" strokeWidth="2"/>
                  <path d="M23 21V19C22.9993 18.1137 22.7044 17.2528 22.1614 16.5523C21.6184 15.8519 20.8581 15.3516 20 15.13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M16 3.13C16.8604 3.35031 17.623 3.85071 18.1676 4.55232C18.7122 5.25392 19.0078 6.11683 19.0078 7.005C19.0078 7.89318 18.7122 8.75608 18.1676 9.45769C17.623 10.1593 16.8604 10.6597 16 10.88" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                User Management
              </a>
            )}
            <a
              href="#receipts"
              className={`nav-item ${currentSection === 'receipts' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('receipts'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <polyline points="10,9 9,9 8,9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Recipes
            </a>
            {runningRecipeId && (
              <a
                href="#realtime"
                className={`nav-item ${currentSection === 'realtime' ? 'active' : ''}`}
                onClick={(e) => { e.preventDefault(); handleSectionChange('realtime'); }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                  <circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="2"/>
                  <circle cx="12" cy="12" r="2" fill="currentColor"/>
                  <circle cx="12" cy="12" r="1" fill="currentColor" opacity="0.5">
                    <animate attributeName="r" from="2" to="10" dur="1.5s" repeatCount="indefinite"/>
                    <animate attributeName="opacity" from="0.5" to="0" dur="1.5s" repeatCount="indefinite"/>
                  </circle>
                </svg>
                Inference Realtime
                <span style={{
                  marginLeft: '0.5rem',
                  padding: '2px 8px',
                  background: 'rgba(34, 197, 94, 0.2)',
                  color: '#22c55e',
                  borderRadius: '8px',
                  fontSize: '0.7rem',
                  fontWeight: 700
                }}>
                  LIVE
                </span>
              </a>
            )}
            {canAccessPage('cameraManagement') && (
              <a
                href="#cameras"
                className={`nav-item ${currentSection === 'cameras' ? 'active' : ''}`}
                onClick={(e) => { e.preventDefault(); handleSectionChange('cameras'); }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <rect x="2" y="4" width="20" height="16" rx="2" stroke="currentColor" strokeWidth="2"/>
                  <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                  <path d="M7 4v2M17 4v2" stroke="currentColor" strokeWidth="2"/>
                </svg>
                Cameras
              </a>
            )}
            <a
              href="#historical"
              className={`nav-item ${currentSection === 'historical' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('historical'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                <polyline points="12,6 12,12 16,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Historical
            </a>
          </nav>

          <h3>Appearance</h3>
          <div className="appearance-controls">
            <div className="dark-mode-toggle">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M21 12.79A9 9 0 1 1 11.21 3A7 7 0 0 0 21 12.79Z" stroke="currentColor" strokeWidth="2"/>
              </svg>
              <span>Dark mode</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={darkMode}
                  onChange={(e) => setDarkMode(e.target.checked)}
                />
                <span className="slider"></span>
              </label>
            </div>
          </div>

          <h3>Action</h3>
          <nav className="action-menu">
            {canAccessPage('logs') && (
              <a
                href="#logs"
                className={`nav-item ${currentSection === 'logs' ? 'active' : ''}`}
                onClick={(e) => { e.preventDefault(); handleSectionChange('logs'); }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M9 12H15M9 16H15M17 21H7C5.89543 21 5 20.1046 5 19V5C5 3.89543 5.89543 3 7 3H12L19 10V19C19 20.1046 18.1046 21 17 21Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 3V10H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Logs
              </a>
            )}
            {canAccessPage('settings') && (
              <a
                href="#settings"
                className={`nav-item ${currentSection === 'settings' ? 'active' : ''}`}
                onClick={(e) => { e.preventDefault(); handleSectionChange('settings'); }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
                  <path d="M19.4 15A1.65 1.65 0 0 0 21 13.35A1.65 1.65 0 0 0 19.4 11.65L18.7 12.35A7.81 7.81 0 0 0 12 5.29V4A1 1 0 0 0 11 3A1 1 0 0 0 10 4V5.29A7.81 7.81 0 0 0 3.3 12.35L2.6 11.65A1.65 1.65 0 0 0 1 13.35A1.65 1.65 0 0 0 2.6 15L3.3 14.3A7.81 7.81 0 0 0 10 20.71V22A1 1 0 0 0 11 23A1 1 0 0 0 12 22V20.71A7.81 7.81 0 0 0 18.7 14.3L19.4 15Z" stroke="currentColor" strokeWidth="2"/>
                </svg>
                Settings
              </a>
            )}
            <a
              href="#documentation"
              className={`nav-item ${currentSection === 'documentation' ? 'active' : ''}`}
              onClick={(e) => { e.preventDefault(); handleSectionChange('documentation'); }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M6.5 2H20V22H6.5A2.5 2.5 0 0 1 4 19.5V4.5A2.5 2.5 0 0 1 6.5 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M8 7H16M8 11H16M8 15H12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Documentation
            </a>
            {canAccessPage('system') && (
              <a
                href="#system"
                className={`nav-item ${currentSection === 'system' ? 'active' : ''}`}
                onClick={(e) => { e.preventDefault(); handleSectionChange('system'); }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <rect x="2" y="3" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="2"/>
                  <path d="M8 21H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M12 17V21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  <circle cx="12" cy="10" r="2" fill="currentColor"/>
                </svg>
                System I/O
              </a>
            )}
            <a href="#logout" className="nav-item" onClick={(e) => { e.preventDefault(); handleLogout(); }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M9 21H5A2 2 0 0 1 3 19V5A2 2 0 0 1 5 3H9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <polyline points="16,17 21,12 16,7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <line x1="21" y1="12" x2="9" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Logout
            </a>
          </nav>
        </aside>

        {/* Main Dashboard */}
        <main className="dashboard-main" style={{ position: 'relative' }}>
          {isLoading ? (
            <div className={`loading-overlay ${pageLoadingOverlayMode === 'dashboard-main' ? 'dashboard-main-only' : ''}`}>
              <div className="loading-container">
                {renderLoadingTemplate(loadingTemplates[currentSection] || 'camera-vision')}
              </div>
            </div>
          ) : (
            <>
              {currentSection === 'dashboard' && (
                <section className="content-section active">
              <div className="dashboard-header-section">
                <div className="welcome-section">
                  <h1>Camera Production Dashboard</h1>
                  <p>Monitor camera performance and product processing in real-time</p>
                </div>
                <div className="time-section">
                  <div className="time">{timeString}</div>
                  <div className="date">{dateString}</div>
                </div>
              </div>

              {/* Summary Cards Row */}
              <div className="summary-cards">
                <div className="summary-card">
                  <div className="card-icon camera">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <rect x="2" y="6" width="20" height="12" rx="2" stroke="currentColor" strokeWidth="2"/>
                      <circle cx="12" cy="12" r="2.5" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Total Cameras</h3>
                    <div className="card-value">{cameraStats.total}</div>
                    <span className={`card-status ${cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? 'online' : 'offline'}`}>
                      {cameraStats.connected === cameraStats.total && cameraStats.total > 0 ? 'All online' : `${cameraStats.connected} online`}
                    </span>
                  </div>
                </div>

                <div className="summary-card">
                  <div className="card-icon products">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <path d="M21 16V8C20.9996 7.64927 20.9071 7.30481 20.7315 7.00116C20.556 6.69751 20.3037 6.44536 20 6.27L13 2.27C12.696 2.09446 12.3511 2.00205 12 2.00205C11.6489 2.00205 11.304 2.09446 11 2.27L4 6.27C3.69626 6.44536 3.44398 6.69751 3.26846 7.00116C3.09294 7.30481 3.00036 7.64927 3 8V16C3.00036 16.3507 3.09294 16.6952 3.26846 16.9988C3.44398 17.3025 3.69626 17.5546 4 17.73L11 21.73C11.304 21.9055 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9055 13 21.73L20 17.73C20.3037 17.5546 20.556 17.3025 20.7315 16.9988C20.9071 16.6952 20.9996 16.3507 21 16Z" stroke="currentColor" strokeWidth="2"/>
                      <polyline points="3.27,6.96 12,12.01 20.73,6.96" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <line x1="12" y1="22.08" x2="12" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Products Today</h3>
                    <div className="card-value">{totalProductsToday.toLocaleString()}</div>
                    <span className={`card-status ${todayStats.changePercent >= 0 ? 'increase' : 'decrease'}`}>
                      {todayStats.changePercent >= 0 ? '+' : ''}{todayStats.changePercent}% vs yesterday
                    </span>
                  </div>
                </div>

                <div className="summary-card">
                  <div className="card-icon processing">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                      <polyline points="12,6 12,12 16,14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Pass Count</h3>
                    <div className="card-value">{todayStats.pass.toLocaleString()}</div>
                    <span className="card-status success">{todayStats.total > 0 ? `${Math.round(todayStats.pass / todayStats.total * 100)}%` : '0%'} of total</span>
                  </div>
                </div>

                <div className="summary-card">
                  <div className="card-icon success">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <polyline points="20,6 9,17 4,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </div>
                  <div className="card-info">
                    <h3>Success Rate</h3>
                    <div className="card-value">{todayStats.passRate.toFixed(1)}%</div>
                    <span className={`card-status ${todayStats.passRate >= 90 ? 'success' : todayStats.passRate >= 70 ? 'normal' : 'offline'}`}>
                      {todayStats.passRate >= 90 ? 'Excellent' : todayStats.passRate >= 70 ? 'Good' : 'Needs attention'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="dashboard-main-content">
                {/* Left Section - Camera Previews */}
                <div className="dashboard-left">
                  <div className="camera-previews">
                    {cameras.map((camera, index) => (
                      <div key={camera.camera_id} className="camera-preview-card">
                        <div className="camera-preview-header">
                          <div className="camera-preview-title">
                            <div className={`camera-status ${camera.is_connected ? 'active' : 'inactive'}`}></div>
                            <span>{camera.camera_id} - {camera.location || camera.model_name}</span>
                          </div>
                          {camera.is_connected && <span className="live-badge">LIVE</span>}
                        </div>
                        <div className="camera-preview-frame">
                          {(() => {
                            const serialNumber = camera.serial_number || camera.camera_id;
                            const frameUrl = cameraFrames[serialNumber];

                            if (!camera.is_connected || !frameUrl) {
                              return (
                                <div style={{
                                  width: '100%',
                                  height: '100%',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  background: '#1f2937',
                                  color: '#9ca3af'
                                }}>
                                  No Display
                                </div>
                              );
                            }

                            return (
                              <img
                                src={frameUrl}
                                alt={`${camera.camera_id} Feed`}
                                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                              />
                            );
                          })()}
                          <div className="camera-overlay-info">
                            <div className="overlay-stat">
                              <span className="overlay-label">FPS:</span>
                              <span className="overlay-value">{camera.max_frame_rate || 30}</span>
                            </div>
                            <div className="overlay-stat">
                              <span className="overlay-label">Resolution:</span>
                              <span className="overlay-value">{camera.resolution_width}x{camera.resolution_height}</span>
                            </div>
                            <div className="overlay-stat">
                              <span className="overlay-label">IP:</span>
                              <span className="overlay-value">{camera.ip_address || 'N/A'}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                    {cameras.length === 0 && (
                      <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280' }}>
                        No cameras available. Please add cameras in Camera Management.
                      </div>
                    )}
                  </div>
                </div>

                {/* Center Section - Recipe Statistics Charts */}
                <div className="dashboard-center">
                  <div className="camera-stats">
                    {recipeCharts.length > 0 ? (
                      <>
                        {recipeCharts.map((recipeChart) => (
                          <div key={recipeChart.recipeId}>
                            <RecipeChart
                              recipeId={recipeChart.recipeId}
                              recipeName={recipeChart.recipeName}
                              timeRange="1h"
                              labels={recipeChart.oneHour.labels}
                              totalData={recipeChart.oneHour.totalData}
                              passData={recipeChart.oneHour.passData}
                              failData={recipeChart.oneHour.failData}
                            />
                            <RecipeChart
                              recipeId={recipeChart.recipeId}
                              recipeName={recipeChart.recipeName}
                              timeRange="today"
                              labels={recipeChart.today.labels}
                              totalData={recipeChart.today.totalData}
                              passData={recipeChart.today.passData}
                              failData={recipeChart.today.failData}
                            />
                            <RecipeChart
                              recipeId={recipeChart.recipeId}
                              recipeName={recipeChart.recipeName}
                              timeRange="month"
                              labels={recipeChart.month.labels}
                              totalData={recipeChart.month.totalData}
                              passData={recipeChart.month.passData}
                              failData={recipeChart.month.failData}
                            />
                          </div>
                        ))}
                      </>
                    ) : (
                      <div style={{ padding: '3rem', textAlign: 'center', color: '#6b7280', background: 'white', borderRadius: '12px' }}>
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 1rem', opacity: 0.3 }}>
                          <path d="M21 16V8C20.9996 7.64927 20.9071 7.30481 20.7315 7.00116C20.556 6.69751 20.3037 6.44536 20 6.27L13 2.27C12.696 2.09446 12.3511 2.00205 12 2.00205C11.6489 2.00205 11.304 2.09446 11 2.27L4 6.27C3.69626 6.44536 3.44398 6.69751 3.26846 7.00116C3.09294 7.30481 3.00036 7.64927 3 8V16C3.00036 16.3507 3.09294 16.6952 3.26846 16.9988C3.44398 17.3025 3.69626 17.5546 4 17.73L11 21.73C11.304 21.9055 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9055 13 21.73L20 17.73C20.3037 17.5546 20.556 17.3025 20.7315 16.9988C20.9071 16.6952 20.9996 16.3507 21 16Z" stroke="currentColor" strokeWidth="2"/>
                        </svg>
                        <p>No products processed today</p>
                        <p style={{ fontSize: '14px', marginTop: '0.5rem' }}>Charts will appear once products are inspected</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Section */}
                <div className="dashboard-right">
                  {/* Recent Members */}
                  <div className="right-card recent-members">
                    <h3>Recent Visited Members</h3>
                      <div className="member-list">
                        {recentMembers.length === 0 ? (
                          <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>
                            No recent members
                          </div>
                        ) : (
                          recentMembers.map((m) => (
                            <div key={m.username} className="member-item">
                              {m.avatar_url! ? (
                                <img
                                  src={`${API_BASE_URL}${m.avatar_url}`}
                                  alt={m.full_name || m.username}
                                  className="member-avatar"
                                  style={{ width: 48, height: 48, borderRadius: 30, objectFit: 'cover' }}
                                />
                              ) : (
                                <div className="member-avatar" style={{ width: 48, height: 48, borderRadius: 30, background: '#e5e7eb', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#374151' }}>
                                  {(m.full_name && m.full_name.charAt(0)) || m.username.charAt(0)}
                                </div>
                              )}
                              <div className="member-info">
                                <h4>{m.username}{m.full_name ? ` — ${m.full_name}` : ''}</h4>
                                <span>{m.last_seen ? `Last seen ${m.last_seen}` : ''}</span>
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                      {currentUser?.role === 'admin' ? (
                        <a
                          href="#"
                          className="view-all"
                          onClick={(e) => { e.preventDefault(); handleSectionChange('users'); }}
                        >
                          View all Members
                        </a>
                      ) : (
                        <span className="view-all" style={{ color: '#9ca3af', cursor: 'default', pointerEvents: 'none' }}>View all Members</span>
                      )}
                  </div>

                  {/* Recent Recipes */}
                  <div className="right-card recent-products">
                    <h3>Recent Recipes</h3>
                    <div className="product-list">
                      {recentLoads.length > 0 ? recentLoads.map((load) => {
                        const localTime = load.loaded_at_local ? (() => {
                          // Parse ISO string manually: 2025-12-22T23:44:33.004945+07:00
                          const match = load.loaded_at_local!.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
                          if (!match) return 'Invalid date';
                          const [, year, month, day, hour, minute] = match;
                          const date = new Date(parseInt(year!), parseInt(month!) - 1, parseInt(day!), parseInt(hour!), parseInt(minute!));
                          return date.toLocaleString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                            hour12: true
                          });
                        })() : 'N/A';
                        return (
                          <div key={load.id} className="product-item">
                            <div className="product-header">
                              <div className="product-icon success">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                                  <polyline points="20,6 9,17 4,12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                              </div>
                              <div className="product-info">
                                <h4>{load.metadata?.name || 'Unknown Recipe'}</h4>
                                <span className="product-line">{load.metadata?.product_code || 'N/A'}</span>
                              </div>
                            </div>
                            <div className="product-details">
                              <div className="detail-row">
                                <span className="detail-label">Loaded by:</span>
                                <span className="detail-value">{load.loaded_by_full_name || load.loaded_by}</span>
                              </div>
                              <div className="detail-row">
                                <span className="detail-label">Loaded at:</span>
                                <span className="detail-value">{localTime}</span>
                              </div>
                              {/* <div className="detail-row">
                                <span className="detail-label">Recipe ID:</span>
                                <span className="detail-value">{load.recipe_id}</span>
                              </div> */}
                            </div>
                          </div>
                        );
                      }) : (
                        <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>
                          No recent recipe loads
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </section>
          )}

          {currentSection === 'users' && <UserManagement />}
          {currentSection === 'receipts' && <Receipts />}
          {currentSection === 'realtime' && <InferenceRealtime runningRecipeId={runningRecipeId} />}
          {currentSection === 'cameras' && <CameraManagement />}
          {currentSection === 'historical' && <Historical />}
          {currentSection === 'logs' && <Logs />}
          {currentSection === 'settings' && <Settings />}
          {currentSection === 'documentation' && <Documentation />}
          {currentSection === 'system' && canAccessPage('system') && <System />}
            </>
          )}
        </main>
      </div>

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onClose={() => setConfirmDialog({ ...confirmDialog, isOpen: false })}
        onConfirm={confirmDialog.onConfirm || (() => {})}
        title={confirmDialog.title}
        message={confirmDialog.message}
        type={confirmDialog.type}
      />

      {/* AI Agent Chat Widget */}
      <AgentChatWidget />
    </div>
  );
}