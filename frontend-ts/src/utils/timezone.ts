/**
 * Timezone Utility Functions
 *
 * Handles conversion between UTC (used by backend) and local timezone (for display)
 * Default timezone: UTC+7 (Vietnam/Bangkok)
 */

export const TIMEZONE_OFFSETS = {
  'UTC+0': 0,
  'UTC+7': 7 * 60,  // Vietnam, Bangkok, Hanoi (minutes)
  'UTC+8': 8 * 60,  // Singapore, Beijing
  'UTC+9': 9 * 60,  // Tokyo, Seoul
} as const;

export type TimezoneKey = keyof typeof TIMEZONE_OFFSETS;

/**
 * Get timezone offset from settings or use default (UTC+7)
 */
export function getTimezoneOffset(): number {
  try {
    const settingsStr = localStorage.getItem('appSettings');
    if (settingsStr) {
      const settings = JSON.parse(settingsStr);
      const timezone = settings.timezone || 'UTC+7 (Bangkok, Hanoi)';

      // Extract UTC+X from string like "UTC+7 (Bangkok, Hanoi)"
      const match = timezone.match(/UTC([+-]\d+)/);
      if (match) {
        const offset = parseInt(match[1]);
        return offset * 60; // Convert hours to minutes
      }
    }
  } catch (error) {
    console.warn('Error reading timezone from settings:', error);
  }

  // Default to UTC+7 (Vietnam)
  return TIMEZONE_OFFSETS['UTC+7'];
}

/**
 * Get timezone key from settings or use default
 */
export function getTimezoneKey(): TimezoneKey {
  try {
    const settingsStr = localStorage.getItem('appSettings');
    if (settingsStr) {
      const settings = JSON.parse(settingsStr);
      const timezone = settings.timezone || 'UTC+7';

      // Extract UTC+X
      const match = timezone.match(/UTC([+-]\d+)/);
      if (match) {
        const key = `UTC${match[1]}` as TimezoneKey;
        if (key in TIMEZONE_OFFSETS) {
          return key;
        }
      }
    }
  } catch (error) {
    console.warn('Error reading timezone key from settings:', error);
  }

  return 'UTC+7';
}

/**
 * Convert local time (as interpreted by user) to UTC
 *
 * IMPORTANT: DatePicker returns Date object in BROWSER timezone, but user thinks it's in VN timezone.
 * We need to "reinterpret" the date/time values as VN timezone.
 *
 * Example:
 * - User picks: "17/01/2026 02:15" thinking it's Vietnam time
 * - DatePicker creates: Date object with year=2026, month=0, day=17, hour=2, minute=15 in BROWSER TZ
 * - We need to interpret those values as "17/01/2026 02:15 VN time" and convert to UTC
 * - Result: "2026-01-16T19:15:00.000Z" (UTC)
 *
 * @param localDate - Date object from DatePicker (in browser timezone)
 * @param timezoneOffset - Target timezone offset in minutes (e.g., 420 for UTC+7)
 * @returns Date object in UTC
 */
export function convertLocalToUTC(localDate: Date, timezoneOffset: number): Date {
  // Extract the date/time COMPONENTS (year, month, day, etc.)
  // These are what the USER sees and thinks are in VN timezone
  const year = localDate.getFullYear();
  const month = localDate.getMonth();
  const day = localDate.getDate();
  const hours = localDate.getHours();
  const minutes = localDate.getMinutes();
  const seconds = localDate.getSeconds();
  const milliseconds = localDate.getMilliseconds();

  // Create UTC timestamp using Date.UTC()
  // This creates a timestamp for "YYYY-MM-DD HH:MM:SS UTC"
  const utcTimestamp = Date.UTC(year, month, day, hours, minutes, seconds, milliseconds);

  // Now subtract the timezone offset to convert "local" → UTC
  // Example: "17/01 02:15 VN" = "17/01 02:15 UTC" - 7 hours = "16/01 19:15 UTC"
  const adjustedTimestamp = utcTimestamp - (timezoneOffset * 60 * 1000);

  return new Date(adjustedTimestamp);
}

/**
 * Convert UTC time to local time
 *
 * Example:
 * - utcDate: "2026-01-16T19:00:00Z" (UTC)
 * - timezoneOffset: 420 (UTC+7)
 * - Returns: Date representing "2026-01-17 02:00" local time
 *
 * @param utcDate - Date object in UTC
 * @param timezoneOffset - Offset in minutes (e.g., 420 for UTC+7)
 * @returns Date object in local timezone
 */
export function convertUTCToLocal(utcDate: Date, timezoneOffset: number): Date {
  // Add offset to UTC time to get local time
  const localTime = utcDate.getTime() + (timezoneOffset * 60 * 1000);
  return new Date(localTime);
}

/**
 * Format a UTC timestamp string to local time string
 *
 * @param utcTimestamp - ISO 8601 timestamp string (e.g., "2026-01-16T19:42:08.059Z")
 * @param format - 'time' | 'date' | 'datetime'
 * @returns Formatted string in local timezone
 */
export function formatTimestamp(
  utcTimestamp: string,
  format: 'time' | 'date' | 'datetime' = 'datetime'
): string {
  const utcDate = new Date(utcTimestamp);
  const offset = getTimezoneOffset();
  const localDate = convertUTCToLocal(utcDate, offset);

  switch (format) {
    case 'time':
      return localDate.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });

    case 'date':
      return localDate.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });

    case 'datetime':
      return localDate.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });

    default:
      return localDate.toLocaleString('en-US');
  }
}

/**
 * Get current date/time in local timezone
 * Useful for initializing date pickers
 */
export function getCurrentLocalTime(): Date {
  const now = new Date();
  const offset = getTimezoneOffset();
  return convertUTCToLocal(now, offset);
}

/**
 * Parse a date range preset (today, yesterday, etc.) and return UTC timestamps
 *
 * @param range - Preset range key
 * @returns Object with start_date and end_date in ISO format (UTC)
 */
export function getDateRangeUTC(range: string): { start_date: string; end_date: string } {
  const offset = getTimezoneOffset();

  // Get current UTC time
  const nowUTC = new Date();

  // Convert to target timezone to get current local date/time
  // We add offset to UTC to get local time
  const localTimestamp = nowUTC.getTime() + (offset * 60 * 1000);
  const localNow = new Date(localTimestamp);

  // Extract local date/time components
  let startYear = localNow.getUTCFullYear();
  let startMonth = localNow.getUTCMonth();
  let startDay = localNow.getUTCDate();
  let startHour = localNow.getUTCHours();
  let startMinute = localNow.getUTCMinutes();
  let startSecond = localNow.getUTCSeconds();

  let endYear = startYear;
  let endMonth = startMonth;
  let endDay = startDay;
  let endHour = startHour;
  let endMinute = startMinute;
  let endSecond = startSecond;

  switch (range) {
    case 'today':
      // Start: 00:00:00 today (local)
      startHour = 0;
      startMinute = 0;
      startSecond = 0;
      // End: 23:59:59 today (local)
      endHour = 23;
      endMinute = 59;
      endSecond = 59;
      break;

    case 'yesterday':
      // Start: 00:00:00 yesterday (local)
      const yesterday = new Date(localTimestamp - 24 * 60 * 60 * 1000);
      startYear = yesterday.getUTCFullYear();
      startMonth = yesterday.getUTCMonth();
      startDay = yesterday.getUTCDate();
      startHour = 0;
      startMinute = 0;
      startSecond = 0;
      // End: 23:59:59 yesterday (local)
      endYear = startYear;
      endMonth = startMonth;
      endDay = startDay;
      endHour = 23;
      endMinute = 59;
      endSecond = 59;
      break;

    case '7days':
      // Start: 7 days ago, same time
      const sevenDaysAgo = new Date(localTimestamp - 7 * 24 * 60 * 60 * 1000);
      startYear = sevenDaysAgo.getUTCFullYear();
      startMonth = sevenDaysAgo.getUTCMonth();
      startDay = sevenDaysAgo.getUTCDate();
      startHour = sevenDaysAgo.getUTCHours();
      startMinute = sevenDaysAgo.getUTCMinutes();
      startSecond = sevenDaysAgo.getUTCSeconds();
      // End: now (keep current values)
      break;

    case 'thisweek':
      // Start: Monday 00:00:00 of this week (local)
      const dayOfWeek = localNow.getUTCDay();
      const daysToMonday = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
      const monday = new Date(localTimestamp - daysToMonday * 24 * 60 * 60 * 1000);
      startYear = monday.getUTCFullYear();
      startMonth = monday.getUTCMonth();
      startDay = monday.getUTCDate();
      startHour = 0;
      startMinute = 0;
      startSecond = 0;
      // End: now (keep current values)
      break;

    case '30days':
      // Start: 30 days ago, same time
      const thirtyDaysAgo = new Date(localTimestamp - 30 * 24 * 60 * 60 * 1000);
      startYear = thirtyDaysAgo.getUTCFullYear();
      startMonth = thirtyDaysAgo.getUTCMonth();
      startDay = thirtyDaysAgo.getUTCDate();
      startHour = thirtyDaysAgo.getUTCHours();
      startMinute = thirtyDaysAgo.getUTCMinutes();
      startSecond = thirtyDaysAgo.getUTCSeconds();
      // End: now (keep current values)
      break;

    case 'thismonth':
      // Start: 1st day of current month, 00:00:00 (local)
      startDay = 1;
      startHour = 0;
      startMinute = 0;
      startSecond = 0;
      // End: now (keep current values)
      break;

    case 'lastmonth':
      // Start: 1st day of last month, 00:00:00 (local)
      const lastMonthDate = new Date(Date.UTC(startYear, startMonth - 1, 1));
      startYear = lastMonthDate.getUTCFullYear();
      startMonth = lastMonthDate.getUTCMonth();
      startDay = 1;
      startHour = 0;
      startMinute = 0;
      startSecond = 0;
      // End: Last day of last month, 23:59:59 (local)
      const lastDayOfLastMonth = new Date(Date.UTC(endYear, endMonth, 0));
      endYear = lastDayOfLastMonth.getUTCFullYear();
      endMonth = lastDayOfLastMonth.getUTCMonth();
      endDay = lastDayOfLastMonth.getUTCDate();
      endHour = 23;
      endMinute = 59;
      endSecond = 59;
      break;

    default:
      // Default: last 7 days
      const defaultStart = new Date(localTimestamp - 7 * 24 * 60 * 60 * 1000);
      startYear = defaultStart.getUTCFullYear();
      startMonth = defaultStart.getUTCMonth();
      startDay = defaultStart.getUTCDate();
      startHour = defaultStart.getUTCHours();
      startMinute = defaultStart.getUTCMinutes();
      startSecond = defaultStart.getUTCSeconds();
  }

  // Convert local date/time components to UTC timestamps
  // Create UTC timestamp, then subtract offset to get actual UTC
  const startUTCTimestamp = Date.UTC(startYear, startMonth, startDay, startHour, startMinute, startSecond) - (offset * 60 * 1000);
  const endUTCTimestamp = Date.UTC(endYear, endMonth, endDay, endHour, endMinute, endSecond) - (offset * 60 * 1000);

  return {
    start_date: new Date(startUTCTimestamp).toISOString(),
    end_date: new Date(endUTCTimestamp).toISOString()
  };
}

/**
 * Get timezone display name
 */
export function getTimezoneDisplay(): string {
  const key = getTimezoneKey();
  const offset = TIMEZONE_OFFSETS[key];
  const hours = offset / 60;
  const sign = hours >= 0 ? '+' : '';
  return `UTC${sign}${hours}`;
}
