/**
 * Jetson System Monitoring Types
 */

export interface CPUMetrics {
  usage_percent: number;
  frequency_mhz: number;
  temperature_celsius: number | null;
  per_core_usage: { [key: string]: number };
}

export interface GPUMetrics {
  usage_percent: number;
  frequency_mhz: number;
  temperature_celsius: number | null;
}

export interface RAMMetrics {
  total_mb: number;
  used_mb: number;
  available_mb: number;
  usage_percent: number;
  swap_total_mb: number;
  swap_used_mb: number;
  swap_percent: number;
}

export interface DiskMetrics {
  total_gb: number;
  used_gb: number;
  available_gb: number;
  usage_percent: number;
  read_speed_mbps: number | null;
  write_speed_mbps: number | null;
}

export interface PowerMetrics {
  total_mw: number | null;
  cpu_gpu_cv_mw: number | null;
  soc_mw: number | null;
}

export interface SystemMetrics {
  timestamp: string;
  cpu: CPUMetrics;
  gpu: GPUMetrics;
  ram: RAMMetrics;
  disk: DiskMetrics;
  power: PowerMetrics | null;
  uptime_seconds: number;
  nvp_model: string | null;
}

export interface Alert {
  timestamp: string;
  alert_type: string;
  severity: 'warning' | 'critical';
  message: string;
  current_value: number;
  threshold_value: number;
}

export interface MonitoringStatus {
  monitoring_enabled: boolean;
  update_interval_seconds: number;
  last_update: string | null;
  active_alerts_count: number;
  task_alive: boolean;
  jtop_available: boolean;
}
