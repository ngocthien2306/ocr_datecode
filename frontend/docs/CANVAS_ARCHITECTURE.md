# Canvas Annotation System - Clean Architecture

## 📁 Cấu trúc thư mục

```
frontend/src/
├── fabric/
│   ├── types.js              # Type definitions và constants
│   ├── utils/
│   │   ├── objectUtils.js    # Canvas object utilities (create, find, manipulate)
│   │   └── drawingUtils.js   # Drawing utilities (rectangle, polygon)
│   └── actions/
│       └── canvasActions.js  # Canvas actions (delete, select, highlight)
├── components/
│   ├── TemplateEditor.jsx            # (Old) Original implementation
│   ├── TemplateEditorNew.jsx         # (Old) React-canvas-annotator attempt
│   └── TemplateEditorRefactored.jsx  # ✅ New clean architecture
└── utils/
    └── annotationAdapter.js  # (Old) Adapter for react-canvas-annotator
```

## ✨ Architecture mới

### 1. **Separation of Concerns**

#### `fabric/types.js`
- Định nghĩa annotation types (text, barcode, template, etc.)
- Constants cho colors, shapes
- Type configurations

#### `fabric/utils/objectUtils.js`
- `createRectangleObject()` - Tạo Fabric Rectangle
- `createPolygonObject()` - Tạo Fabric Polygon
- `createLabel()` - Tạo text label
- `createPolygonEditPoint()` - Tạo edit point cho polygon
- `findObjectByIndex()` - Tìm object theo index
- `getRectangleData()` / `getPolygonData()` - Extract data từ Fabric objects
- `setupCanvasPanning()` - Setup pan với Space/Shift/Middle-click
- `setCanvasZoom()` / `resetCanvasZoom()` - Zoom controls

#### `fabric/utils/drawingUtils.js`
- `startDrawingRectangle()` - Rectangle drawing với callback
- `PolygonDrawer` class - Quản lý polygon drawing state
  - `addPoint()` - Thêm điểm
  - `complete()` - Hoàn thành polygon
  - `cancel()` - Hủy vẽ
  - `getRemainingPoints()` - Số điểm còn lại

#### `fabric/actions/canvasActions.js`
- `deleteSelected()` - Xóa selected objects
- `deleteByIndex()` - Xóa theo annotation index
- `clearDrawingHelpers()` - Clear temp objects
- `highlightObject()` - Highlight object
- `deselectAll()` - Deselect tất cả

### 2. **Component Architecture**

```javascript
TemplateEditorRefactored.jsx
├── State Management
│   ├── drawMode (select | polygon)
│   ├── isSpacePressed (for panning)
│   ├── showHints (toggle hints panel)
│   └── polygonEditPoints (array of edit circles)
│
├── Canvas Lifecycle
│   ├── Initialize canvas
│   ├── Setup panning
│   ├── Load background image
│   └── Event handlers (selection, modification)
│
├── Drawing System
│   ├── PolygonDrawer instance
│   ├── Rectangle drawer (future)
│   └── Drawing helpers (temp lines, circles)
│
└── Annotation Management
    ├── loadAnnotations() - Render annotations to canvas
    ├── updateAnnotationFromObject() - Sync Fabric → State
    ├── addAnnotation() - Create new annotation
    └── deleteAnnotation() - Remove annotation
```

## 🎯 Key Features

### ✅ Implemented
- ✨ Polygon drawing (4 points)
- 🎨 Color-coded annotation types
- 🔍 Zoom & Pan (Space/Shift/Middle-click + drag)
- ✏️ Polygon editing với edit points
- ⌨️ Keyboard shortcuts (V, P, Esc, Del, H)
- 🏷️ Type labels trên annotations
- 💡 CVAT-style hints panel
- 🗑️ Delete selected/by index

### 🚧 Future Enhancements
- Rectangle drawing tool
- Rotation support
- Copy/paste annotations
- Undo/redo
- Export annotations
- Import from COCO/YOLO format

## 📝 Usage

### Basic Usage

```javascript
import TemplateEditor from './components/TemplateEditorRefactored';

<TemplateEditor
  templateImage={imageUrl}
  annotations={annotations}
  onAnnotationsChange={setAnnotations}
  selectedAnnotation={selectedIndex}
  onSelectAnnotation={setSelectedIndex}
/>
```

### Annotation Format

```javascript
{
  id: "annotation-123",
  type: "text",        // text | barcode | template | crop_area | datecode
  shape: "polygon",    // rectangle | polygon
  points: [            // For polygon
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
  ],
  text: "Sample text" // Optional text content
}
```

### Adding Custom Utilities

```javascript
// fabric/utils/yourUtils.js
export const yourFunction = (canvas, params) => {
  // Your logic
};

// Use in component
import * as yourUtils from '../fabric/utils/yourUtils';
yourUtils.yourFunction(canvas, params);
```

## 🔄 Migration Path

### From TemplateEditor.jsx (Old)
```diff
- import TemplateEditor from './TemplateEditor';
+ import TemplateEditor from './TemplateEditorRefactored';
```

### Benefits
- ✅ **Cleaner code** - Logic tách ra utils/actions
- ✅ **Easier testing** - Functions có thể test riêng
- ✅ **Better maintainability** - Tìm và fix bugs dễ hơn
- ✅ **Reusable utilities** - Dùng lại ở component khác
- ✅ **No external dependencies** - Không cần react-canvas-annotator

## 🐛 Known Issues

1. **Rectangle tool** - Chưa implement (dễ thêm với drawingUtils)
2. **Edit points persistence** - Edit points biến mất khi deselect (by design)
3. **Zoom với edit points** - Cần handle scaling cho edit points

## 💡 Tips

### Debugging
```javascript
// Log canvas objects
console.log(canvas.getObjects());

// Find annotation by index
const obj = objectUtils.findObjectByIndex(canvas, 0);

// Get all annotations
const objs = objectUtils.getAnnotationObjects(canvas);
```

### Performance
- Use `objectCaching: false` cho real-time updates
- Batch canvas operations before `requestRenderAll()`
- Remove unused event listeners

### Best Practices
- Always check canvas exists before operations
- Clean up event listeners in useEffect cleanup
- Keep annotation state as single source of truth
- Use utils/actions instead of inline Fabric code

## 📚 References

- [Fabric.js Documentation](http://fabricjs.com/)
- [React Hooks Best Practices](https://react.dev/reference/react)
- Inspired by [react-canvas-annotator](https://github.com/dansreis/react-canvas-annotator)
- Inspired by [CVAT](https://github.com/opencv/cvat)

---

**Created:** December 14, 2025  
**Status:** ✅ Production Ready  
**No External Dependencies:** Pure Fabric.js implementation
