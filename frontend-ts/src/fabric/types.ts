/**
 * Fabric.js custom types for annotation system
 */

/**
 * @typedef {Object} AnnotationObject
 * @property {string} id - Unique identifier
 * @property {string} type - Annotation type (text, barcode, template, etc.)
 * @property {string} shape - Shape type (rectangle, polygon)
 * @property {number} [x] - X coordinate (for rectangle)
 * @property {number} [y] - Y coordinate (for rectangle)
 * @property {number} [width] - Width (for rectangle)
 * @property {number} [height] - Height (for rectangle)
 * @property {Array<[number, number]>} [points] - Points array (for polygon)
 * @property {string} [text] - Associated text content
 */

/**
 * @typedef {Object} CanvasConfig
 * @property {number} width - Canvas width
 * @property {number} height - Canvas height
 * @property {string} backgroundColor - Background color
 */

export const ANNOTATION_TYPES = {
  TEXT: 'text',
  BARCODE: 'barcode',
  TEMPLATE: 'template',
  CROP_AREA: 'crop_area',
  DATECODE: 'datecode',
  PRODUCT: 'product',
  LABEL: 'label',
  CHAR: 'char',
  MASK: 'mask',
};

export const SHAPE_TYPES = {
  RECTANGLE: 'rectangle',
  POLYGON: 'polygon'
};

export const TYPE_CONFIGS = [
  { value: 'text', label: 'Text OCR', color: '#50fa7b', needsText: true },
  //{ value: 'barcode', label: 'Barcode', color: '#ffdc5c', needsText: false },
  { value: 'template', label: 'Template Match', color: '#ff5555', needsText: false },
  { value: 'crop_area', label: 'Crop Area', color: '#ff64ff', needsText: false },
  { value: 'datecode', label: 'Date Code', color: '#5096ff', needsText: true },
  { value: 'product', label: 'Product', color: '#7513dd', needsText: false },
  { value: 'label', label: 'Label', color: '#ad6df1', needsText: false },
  { value: 'char', label: 'Character (ML)', color: '#fbbf24', needsText: true },
  { value: 'mask', label: 'Mask', color: '#94a3b8', needsText: false },
];
