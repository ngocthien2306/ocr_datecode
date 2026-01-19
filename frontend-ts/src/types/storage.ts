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
