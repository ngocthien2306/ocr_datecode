import React, { useRef, useEffect } from 'react';
import { Canvas, FabricImage, Rect, Polygon, Text as FabricText } from 'fabric';
import type { Annotation } from '@/types';
import '@/styles/InferenceTemplateViewer.css';

interface InferenceTemplateViewerProps {
  imageUrl: string;
  annotations: Annotation[];
  selectedAnnotation: number | null;
  onSelectAnnotation: (index: number | null) => void;
}

// Annotation type colors
const ANNOTATION_COLORS: Record<string, string> = {
  text: '#50fa7b',
  barcode: '#ffdc5c',
  template: '#ff5555',
  crop_area: '#ff64ff',
  datecode: '#5096ff',
  product: '#7513dd',
  label: '#ad6df1'
};

export default function InferenceTemplateViewer({
  imageUrl,
  annotations,
  selectedAnnotation,
  onSelectAnnotation
}: InferenceTemplateViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricCanvasRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize canvas
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    console.log('🎨 Initializing InferenceTemplateViewer canvas...');

    const canvas = new Canvas(canvasRef.current, {
      width: 800,
      height: 600,
      backgroundColor: '#1e1e1e',
      selection: false // No selection box
    });

    fabricCanvasRef.current = canvas;

    // Cleanup
    return () => {
      console.log('🧹 Cleaning up InferenceTemplateViewer canvas');
      canvas.dispose();
      fabricCanvasRef.current = null;
    };
  }, []);

  // Load image and annotations
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas || !imageUrl) {
      console.log('⏭️ Skipping render - no canvas or image');
      return;
    }

    console.log('🖼️ Loading image:', imageUrl);

    // Clear canvas
    canvas.clear();
    canvas.backgroundColor = '#1e1e1e';

    // Load background image
    FabricImage.fromURL(imageUrl, { crossOrigin: 'anonymous' })
      .then((img) => {
        if (!img) {
          console.error('❌ Failed to load image');
          return;
        }

        console.log('✅ Image loaded:', img.width, 'x', img.height);

        // Calculate scale to fit canvas
        const scale = Math.min(
          (canvas.width - 100) / img.width,
          (canvas.height - 100) / img.height,
          1 // Don't scale up
        );

        img.scale(scale);
        const imgLeft = (canvas.width - img.width * scale) / 2;
        const imgTop = (canvas.height - img.height * scale) / 2;

        img.set({
          left: imgLeft,
          top: imgTop,
          selectable: false,
          evented: false
        });

        canvas.backgroundImage = img;
        canvas.requestRenderAll();

        // Store image bounds for annotation rendering
        (canvas as any).imageBounds = {
          left: imgLeft,
          top: imgTop,
          width: img.width * scale,
          height: img.height * scale
        };

        console.log('📐 Image bounds:', (canvas as any).imageBounds);

        // Render annotations
        renderAnnotations(canvas, annotations, selectedAnnotation);
      })
      .catch((err) => {
        console.error('❌ Error loading image:', err);
      });
  }, [imageUrl, annotations, selectedAnnotation]);

  const renderAnnotations = (canvas: any, annotations: Annotation[], selectedIndex: number | null) => {
    const bounds = (canvas as any).imageBounds;
    if (!bounds) {
      console.log('⚠️ No image bounds, skipping annotations');
      return;
    }

    console.log('✏️ Rendering', annotations.length, 'annotations');

    // Remove old annotations (keep only background image)
    const objects = canvas.getObjects();
    objects.forEach((obj: any) => {
      if (obj.annotationIndex !== undefined || obj.isLabel) {
        canvas.remove(obj);
      }
    });

    // Add annotations
    annotations.forEach((ann, index) => {
      const isSelected = index === selectedIndex;
      const color = ANNOTATION_COLORS[ann.type || 'text'] || '#50fa7b';
      const strokeColor = isSelected ? '#3b82f6' : color; // Blue when selected
      const strokeWidth = isSelected ? 3 : 2;

      if (ann.shape === 'rectangle' && ann.x !== undefined && ann.y !== undefined && ann.width && ann.height) {
        // Rectangle
        const rect = new Rect({
          left: bounds.left + ann.x * bounds.width,
          top: bounds.top + ann.y * bounds.height,
          width: ann.width * bounds.width,
          height: ann.height * bounds.height,
          fill: 'transparent',
          stroke: strokeColor,
          strokeWidth: strokeWidth,
          selectable: false, // Read-only
          evented: true, // Can click
          hoverCursor: 'pointer'
        });

        (rect as any).annotationIndex = index;
        (rect as any).annotationType = ann.type;

        // Click to select
        rect.on('mousedown', () => {
          console.log('📌 Annotation clicked:', index);
          onSelectAnnotation(index);
        });

        canvas.add(rect);

        // Add text label
        if (ann.text) {
          const label = new FabricText(ann.text, {
            left: bounds.left + ann.x * bounds.width,
            top: bounds.top + ann.y * bounds.height - 20,
            fontSize: 14,
            fill: strokeColor,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            selectable: false,
            evented: false
          });

          (label as any).isLabel = true;
          (label as any).annotationIndex = index;
          canvas.add(label);
        }
      } else if (ann.shape === 'polygon' && ann.points && ann.points.length > 0) {
        // Polygon
        const scaledPoints = ann.points.map(([x, y]) => ({
          x: bounds.left + x * bounds.width,
          y: bounds.top + y * bounds.height
        }));

        const polygon = new Polygon(scaledPoints, {
          fill: 'transparent',
          stroke: strokeColor,
          strokeWidth: strokeWidth,
          selectable: false,
          evented: true,
          hoverCursor: 'pointer'
        });

        (polygon as any).annotationIndex = index;
        (polygon as any).annotationType = ann.type;

        polygon.on('mousedown', () => {
          console.log('📌 Annotation clicked:', index);
          onSelectAnnotation(index);
        });

        canvas.add(polygon);
      }
    });

    canvas.requestRenderAll();
    console.log('✅ Annotations rendered');
  };

  return (
    <div ref={containerRef} className="inference-template-viewer">
      <canvas ref={canvasRef} />
    </div>
  );
}
