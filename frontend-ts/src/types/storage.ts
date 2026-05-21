export interface StorageItem {
  name: string;
  path: string;
  type: 'directory' | 'file';
  size_bytes: number;
  size_mb: number;
  file_count?: number;
  children?: StorageItem[] | null;
  modified_time?: string;
}

export interface StorageStats {
  total_size_bytes: number;
  total_size_mb: number;
  total_size_gb: number;
  total_files: number;
  total_directories: number;
}

export interface DiskUsage {
  total_gb: number;
  used_gb: number;
  free_gb: number;
  used_percent: number;
  mount_point: string;
}

export type CleanupScheduleMode = 'daily' | 'hourly';

export interface CleanupConfig {
  enabled: boolean;
  dry_run: boolean;
  schedule_mode: CleanupScheduleMode;
  schedule_hour: number;
  schedule_minute: number;
  interval_hours: number;
  trigger_usage_percent: number;
  target_usage_percent: number;
  min_images_per_folder: number;
  keep_recent_dates_per_recipe: number;
}

export type CleanupTrigger = 'scheduled' | 'manual' | 'threshold-skip';

export interface CleanupRunResult {
  ran_at: string;
  trigger: CleanupTrigger;
  dry_run: boolean;
  disk_used_percent_before: number;
  disk_used_percent_after: number;
  files_deleted: number;
  folders_deleted: number;
  freed_mb: number;
  leaf_folders_scanned: number;
  duration_seconds: number;
  message: string;
}
