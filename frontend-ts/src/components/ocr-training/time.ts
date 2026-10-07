/** The training services store created_at as naive UTC (datetime.utcnow()) and
 *  serialize it without a zone suffix, which `new Date()` would read as LOCAL
 *  time — 7 h off here. Treat zone-less strings as UTC and show factory time. */
export function fmtServiceTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const hasZone = /(?:[zZ]|[+-]\d{2}:?\d{2})$/.test(iso);
  const d = new Date(hasZone ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('vi-VN', { timeZone: 'Asia/Ho_Chi_Minh', hour12: false });
}
