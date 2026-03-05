// =============================================================================
// recipeValidation.ts
// Pure TypeScript validation utilities — no React dependencies.
// Mirrors backend Pydantic constraints in backend/app/schemas/recipe.py.
// =============================================================================

// ---------------------------------------------------------------------------
// Constraints — single source of truth, keep in sync with backend schemas
// ---------------------------------------------------------------------------

export const RECIPE_CONSTRAINTS = {
  name:            { minLength: 1, maxLength: 100 },
  product_code:    { minLength: 1, maxLength: 50  },
  description:     { maxLength: 500 },
  do_reject_number:{ min: 0, max: 3 },
  // do_alarm_number allows -1 (frontend "None" sentinel) — not range-validated
  normal_pulse_ms: { min: 0, max: 999999 },
} as const

export const CAMERA_CONSTRAINTS = {
  exposure_time:  { min: 0, exclusive: true  }, // > 0
  delay_trigger:  { min: 0, exclusive: false }, // >= 0
  delay_interval: { min: 0, exclusive: false },
  gain:           { min: 0, exclusive: false },
  di_number:      { min: 0, max: 3 },
} as const

export const MODEL_CONSTRAINTS = {
  detection_threshold:   { min: 0.0, max: 1.0 },
  recognition_threshold: { min: 0.0, max: 1.0 },
  matching_threshold:    { min: 0.0, max: 1.0 },
  min_text_size:         { min: 1 },
  max_text_size:         { min: 1 },
} as const

// ---------------------------------------------------------------------------
// Error messages — reference constants so they stay in sync automatically
// ---------------------------------------------------------------------------

export const RECIPE_ERRORS = {
  name: {
    required: 'Recipe name is required',
    maxLength: `Recipe name cannot exceed ${RECIPE_CONSTRAINTS.name.maxLength} characters`,
  },
  product_code: {
    required: 'Product code is required',
    maxLength: `Product code cannot exceed ${RECIPE_CONSTRAINTS.product_code.maxLength} characters`,
  },
  description: {
    maxLength: `Description cannot exceed ${RECIPE_CONSTRAINTS.description.maxLength} characters`,
  },
  do_reject_number: {
    range: `Reject DO number must be between ${RECIPE_CONSTRAINTS.do_reject_number.min} and ${RECIPE_CONSTRAINTS.do_reject_number.max}`,
  },
  normal_pulse_ms: {
    range: `Normal pulse must be between ${RECIPE_CONSTRAINTS.normal_pulse_ms.min} and ${RECIPE_CONSTRAINTS.normal_pulse_ms.max} ms`,
  },
} as const

export const CAMERA_ERRORS = {
  exposure_time:  { required: 'Exposure time is required', range: 'Exposure time must be greater than 0' },
  delay_trigger:  { required: 'Delay trigger is required', range: 'Delay trigger cannot be negative' },
  delay_interval: { range: 'Delay interval cannot be negative' },
  gain:           { required: 'Gain is required', range: 'Gain cannot be negative' },
} as const

export const MODEL_ERRORS = {
  detection_threshold:   { required: 'Detection threshold is required', range: 'Must be between 0.0 and 1.0' },
  recognition_threshold: { required: 'Recognition threshold is required', range: 'Must be between 0.0 and 1.0' },
  matching_threshold:    { required: 'Matching threshold is required', range: 'Must be between 0.0 and 1.0' },
  min_text_size:         { range: 'Min text size must be at least 1' },
  max_text_size:         { range: 'Max text size must be at least 1', crossField: 'Max text size must be >= min text size' },
} as const

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ValidationMode = 'create' | 'edit'

export interface BasicInfoInput {
  name?: string
  product_code?: string
  description?: string
  do_reject_number?: number
  normal_pulse_ms?: number
}

export interface CameraConfigInput {
  camera_id: string
  exposure_time?: number
  delay_trigger?: number
  delay_interval?: number
  gain?: number
}

export interface ModelThresholdsInput {
  detection_threshold?: number
  recognition_threshold?: number
  matching_threshold?: number
  min_text_size?: number
  max_text_size?: number
}

export interface RecipeFormInput {
  basicInfo: BasicInfoInput
  cameras: CameraConfigInput[]
  modelThresholds: ModelThresholdsInput
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Returns true when the errors record contains at least one error. */
export function hasValidationErrors(errors: Record<string, string>): boolean {
  return Object.keys(errors).length > 0
}

// In 'edit' mode a field with value undefined means "not being updated" — skip it.
// A field present (even empty string) is validated.
function shouldValidate(value: unknown, mode: ValidationMode): boolean {
  if (mode === 'create') return true
  return value !== undefined
}

/** Returns true if v is a finite number (not NaN, not Infinity, not a string). */
function isValidNumber(v: unknown): v is number {
  return typeof v === 'number' && isFinite(v) && !isNaN(v)
}

// ---------------------------------------------------------------------------
// validateBasicInfo
// Error keys: name, product_code, description, do_reject_number, normal_pulse_ms
// ---------------------------------------------------------------------------

export function validateBasicInfo(
  data: BasicInfoInput,
  mode: ValidationMode
): Record<string, string> {
  const errors: Record<string, string> = {}

  // name
  if (shouldValidate(data.name, mode)) {
    const name = (data.name ?? '').trim()
    if (!name) {
      errors.name = RECIPE_ERRORS.name.required
    } else if (name.length > RECIPE_CONSTRAINTS.name.maxLength) {
      errors.name = RECIPE_ERRORS.name.maxLength
    }
  }

  // product_code
  if (shouldValidate(data.product_code, mode)) {
    const code = (data.product_code ?? '').trim()
    if (!code) {
      errors.product_code = RECIPE_ERRORS.product_code.required
    } else if (code.length > RECIPE_CONSTRAINTS.product_code.maxLength) {
      errors.product_code = RECIPE_ERRORS.product_code.maxLength
    }
  }

  // description (optional — only check max length)
  if (shouldValidate(data.description, mode) && data.description !== undefined) {
    if (data.description.length > RECIPE_CONSTRAINTS.description.maxLength) {
      errors.description = RECIPE_ERRORS.description.maxLength
    }
  }

  // do_reject_number (select is constrained 0-3, validate defensively)
  if (shouldValidate(data.do_reject_number, mode) && data.do_reject_number !== undefined) {
    const v = data.do_reject_number
    if (!isValidNumber(v) || v < RECIPE_CONSTRAINTS.do_reject_number.min || v > RECIPE_CONSTRAINTS.do_reject_number.max) {
      errors.do_reject_number = RECIPE_ERRORS.do_reject_number.range
    }
  }

  // normal_pulse_ms
  if (shouldValidate(data.normal_pulse_ms, mode) && data.normal_pulse_ms !== undefined) {
    const v = data.normal_pulse_ms
    if (!isValidNumber(v) || v < RECIPE_CONSTRAINTS.normal_pulse_ms.min || v > RECIPE_CONSTRAINTS.normal_pulse_ms.max) {
      errors.normal_pulse_ms = RECIPE_ERRORS.normal_pulse_ms.range
    }
  }

  return errors
}

// ---------------------------------------------------------------------------
// validateCameraConfig
// Error keys: cameras.{index}.exposure_time | delay_trigger | delay_interval | gain
// ---------------------------------------------------------------------------

export function validateCameraConfig(
  camera: CameraConfigInput,
  index: number,
  mode: ValidationMode
): Record<string, string> {
  const errors: Record<string, string> = {}
  const prefix = `cameras.${index}`

  // exposure_time (required in create, > 0)
  if (shouldValidate(camera.exposure_time, mode)) {
    if (!isValidNumber(camera.exposure_time)) {
      errors[`${prefix}.exposure_time`] = CAMERA_ERRORS.exposure_time.required
    } else if (camera.exposure_time <= CAMERA_CONSTRAINTS.exposure_time.min) {
      errors[`${prefix}.exposure_time`] = CAMERA_ERRORS.exposure_time.range
    }
  }

  // delay_trigger (required in create, >= 0)
  if (shouldValidate(camera.delay_trigger, mode)) {
    if (!isValidNumber(camera.delay_trigger)) {
      errors[`${prefix}.delay_trigger`] = CAMERA_ERRORS.delay_trigger.required
    } else if (camera.delay_trigger < CAMERA_CONSTRAINTS.delay_trigger.min) {
      errors[`${prefix}.delay_trigger`] = CAMERA_ERRORS.delay_trigger.range
    }
  }

  // delay_interval (optional, >= 0)
  if (shouldValidate(camera.delay_interval, mode) && camera.delay_interval !== undefined) {
    if (!isValidNumber(camera.delay_interval) || camera.delay_interval < CAMERA_CONSTRAINTS.delay_interval.min) {
      errors[`${prefix}.delay_interval`] = CAMERA_ERRORS.delay_interval.range
    }
  }

  // gain (required in create, >= 0)
  if (shouldValidate(camera.gain, mode)) {
    if (!isValidNumber(camera.gain)) {
      errors[`${prefix}.gain`] = CAMERA_ERRORS.gain.required
    } else if (camera.gain < CAMERA_CONSTRAINTS.gain.min) {
      errors[`${prefix}.gain`] = CAMERA_ERRORS.gain.range
    }
  }

  return errors
}

// ---------------------------------------------------------------------------
// validateCameras
// Merges per-camera errors; in create mode also checks array is non-empty.
// Top-level empty error key: 'cameras'
// ---------------------------------------------------------------------------

export function validateCameras(
  cameras: CameraConfigInput[],
  mode: ValidationMode
): Record<string, string> {
  const errors: Record<string, string> = {}

  if (mode === 'create' && cameras.length === 0) {
    errors.cameras = 'At least one camera must be configured'
  }

  cameras.forEach((cam, index) => {
    Object.assign(errors, validateCameraConfig(cam, index, mode))
  })

  return errors
}

// ---------------------------------------------------------------------------
// validateModelThresholds
// Error keys: model_thresholds.{field}
// Cross-field: max_text_size >= min_text_size
// ---------------------------------------------------------------------------

export function validateModelThresholds(
  data: ModelThresholdsInput,
  mode: ValidationMode
): Record<string, string> {
  const errors: Record<string, string> = {}
  const prefix = 'model_thresholds'

  const checkThreshold = (
    field: 'detection_threshold' | 'recognition_threshold' | 'matching_threshold',
    msgs: { required: string; range: string }
  ) => {
    const v = data[field]
    if (shouldValidate(v, mode)) {
      if (!isValidNumber(v)) {
        errors[`${prefix}.${field}`] = msgs.required
      } else if (v < MODEL_CONSTRAINTS[field].min || v > MODEL_CONSTRAINTS[field].max) {
        errors[`${prefix}.${field}`] = msgs.range
      }
    }
  }

  checkThreshold('detection_threshold',   MODEL_ERRORS.detection_threshold)
  checkThreshold('recognition_threshold', MODEL_ERRORS.recognition_threshold)
  checkThreshold('matching_threshold',    MODEL_ERRORS.matching_threshold)

  // min_text_size
  if (shouldValidate(data.min_text_size, mode) && data.min_text_size !== undefined) {
    if (!isValidNumber(data.min_text_size) || data.min_text_size < MODEL_CONSTRAINTS.min_text_size.min) {
      errors[`${prefix}.min_text_size`] = MODEL_ERRORS.min_text_size.range
    }
  }

  // max_text_size + cross-field check
  if (shouldValidate(data.max_text_size, mode) && data.max_text_size !== undefined) {
    if (!isValidNumber(data.max_text_size) || data.max_text_size < MODEL_CONSTRAINTS.max_text_size.min) {
      errors[`${prefix}.max_text_size`] = MODEL_ERRORS.max_text_size.range
    } else if (data.min_text_size !== undefined && data.max_text_size < data.min_text_size) {
      errors[`${prefix}.max_text_size`] = MODEL_ERRORS.max_text_size.crossField
    }
  }

  return errors
}

// ---------------------------------------------------------------------------
// validateRecipeForm — master function, combines all validators
// ---------------------------------------------------------------------------

export function validateRecipeForm(
  data: RecipeFormInput,
  mode: ValidationMode
): Record<string, string> {
  return {
    ...validateBasicInfo(data.basicInfo, mode),
    ...validateCameras(data.cameras, mode),
    ...validateModelThresholds(data.modelThresholds, mode),
  }
}

// ---------------------------------------------------------------------------
// validateRealtimeSettings — for InferenceRealtimeSettingsModal
// Validates the subset of fields editable in realtime (not saved to DB).
// Error keys: delay_reject, reject_pulse, cameras.{index}.{field}
// ---------------------------------------------------------------------------

export interface RealtimeSettingsInput {
  delay_reject?: number
  reject_pulse?: number
  cameras: CameraConfigInput[]
}

// ---------------------------------------------------------------------------
// parseApiErrors — convert Pydantic 422 detail array to our flat error schema
//
// Pydantic error shape:
//   { detail: [{ loc: ["body", "normal_pulse_ms"], msg: "...", type: "..." }] }
//
// Mapping:
//   ["body", "normal_pulse_ms"]              → "normal_pulse_ms"
//   ["body", "cameras", 0, "exposure_time"]  → "cameras.0.exposure_time"
//   ["body", "model_thresholds", "detection_threshold"] → "model_thresholds.detection_threshold"
//
// Returns:
//   { fieldErrors: Record<string, string>, unhandled: string[] }
//   - fieldErrors: errors we can show inline next to fields
//   - unhandled:   errors with unknown field paths (show as toast)
// ---------------------------------------------------------------------------

interface PydanticError {
  loc: (string | number)[]
  msg: string
  type: string
}

export interface ParsedApiErrors {
  fieldErrors: Record<string, string>
  unhandled: string[]
}

export function parseApiErrors(detail: unknown): ParsedApiErrors {
  const fieldErrors: Record<string, string> = {}
  const unhandled: string[] = []

  if (!Array.isArray(detail)) {
    // Non-422 format: treat the whole thing as a generic message
    const msg = typeof detail === 'string' ? detail : 'An error occurred'
    unhandled.push(msg)
    return { fieldErrors, unhandled }
  }

  for (const err of detail as PydanticError[]) {
    const loc = err.loc ?? []
    // loc[0] is always "body" — strip it
    const fieldPath = loc.slice(1).join('.')

    if (!fieldPath) {
      unhandled.push(err.msg)
      continue
    }

    // Map to our error key schema
    // "cameras.0.exposure_time" matches our naming directly ✓
    // "model_thresholds.detection_threshold" matches directly ✓
    fieldErrors[fieldPath] = err.msg
  }

  return { fieldErrors, unhandled }
}

export function validateRealtimeSettings(data: RealtimeSettingsInput): Record<string, string> {
  const errors: Record<string, string> = {}

  if (data.delay_reject !== undefined && data.delay_reject < 0) {
    errors.delay_reject = 'Delay reject cannot be negative'
  }
  if (data.reject_pulse !== undefined && data.reject_pulse < 0) {
    errors.reject_pulse = 'Reject pulse cannot be negative'
  }

  // Camera fields — validate all present cameras (always 'edit' semantics for realtime)
  data.cameras.forEach((cam, index) => {
    Object.assign(errors, validateCameraConfig(cam, index, 'edit'))
  })

  return errors
}
