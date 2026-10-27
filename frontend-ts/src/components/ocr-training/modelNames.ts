/** Plain names for OCR model families, architectures and starting weights.
 *  Operators never see engine names (SMTR, SVTR, PaddleOCR, PP-OCRvN); the ids
 *  sent to the services are unchanged. */

export type ModelFamily = 'smtr' | 'paddle';

export const FAMILY_LABEL: Record<ModelFamily, string> = {
  smtr: 'Custom · Large',
  paddle: 'Custom · Small',
};

const SIZE_LABEL: Record<string, string> = {
  tiny: 'Tiny', small: 'Small', medium: 'Medium', mobile: 'Light', server: 'Heavy',
};

/** "PP-OCRv6_small_rec" / "PP-OCRv6 small (5.2M)" → "Version 6 · Small". */
export function friendlyArch(s: string): string {
  return s.replace(
    /PP-OCRv(\d+)[ _](tiny|small|medium|mobile|server)(_rec)?(\s*\([^)]*\))?/gi,
    (_m, v: string, size: string) => `Version ${v} · ${SIZE_LABEL[size.toLowerCase()] ?? size}`,
  );
}

const BUILTIN_LABEL: Record<string, string> = {
  general: 'Starter · General text',
  datecode_2406: 'Starter · Date codes',
};

/** Base a run started from: "builtin:general", "pretrained:PP-OCRv6_small_rec", … */
export function friendlyBase(label: string): string {
  if (label.startsWith('builtin:')) {
    const id = label.slice('builtin:'.length);
    return BUILTIN_LABEL[id] ?? `Starter · ${id}`;
  }
  if (label.startsWith('pretrained:')) {
    return `Starter · ${friendlyArch(label.slice('pretrained:'.length))}`;
  }
  return friendlyArch(label);
}

export function friendlyRuntime(runtime: string): string {
  return runtime === 'onnx' ? 'not GPU-optimized' : 'GPU-optimized';
}
