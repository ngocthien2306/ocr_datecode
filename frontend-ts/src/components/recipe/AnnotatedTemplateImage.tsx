import React, { useRef, useEffect, useState } from 'react';
import { TYPE_CONFIGS } from '@/fabric/types';
import { API_BASE_URL } from '@/config/api';

interface Annotation {
  type: string;
  shape: string;
  text?: string;
  conf: number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  points?: Array<[number, number]> | null;
}

interface AnnotatedTemplateImageProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  annotations: Annotation[];
  onImageClick?: () => void;
}

const AnnotatedTemplateImage: React.FC<AnnotatedTemplateImageProps> = ({
  imageUrl,
  imageWidth,
  imageHeight,
  annotations,
  onImageClick
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getColorForType = (type: string): string => {
    const config = TYPE_CONFIGS.find(c => c.value === type);
    return config?.color || '#ffffff';
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      // Calculate display size (max 600px width, maintain aspect ratio)
      const maxWidth = 600;
      const scale = Math.min(maxWidth / imageWidth, 1);
      const displayWidth = imageWidth * scale;
      const displayHeight = imageHeight * scale;

      canvas.width = displayWidth;
      canvas.height = displayHeight;

      // Draw image
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

      // Draw annotations
      annotations.forEach((annotation, index) => {
        const color = getColorForType(annotation.type);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.fillStyle = color + '33'; // 20% opacity for fill

        if (annotation.shape === 'rectangle' && annotation.x !== undefined) {
          // Convert normalized coordinates to canvas coordinates
          const x = annotation.x * displayWidth;
          const y = annotation.y! * displayHeight;
          const width = annotation.width! * displayWidth;
          const height = annotation.height! * displayHeight;

          // Draw rectangle
          ctx.strokeRect(x, y, width, height);
          ctx.fillRect(x, y, width, height);

          // Draw label
          const label = `${index + 1}`;
          ctx.fillStyle = color;
          ctx.font = 'bold 14px Arial';
          ctx.fillText(label, x + 4, y + 16);
        } else if (annotation.shape === 'polygon' && annotation.points && annotation.points.length > 0) {
          // Draw polygon
          ctx.beginPath();
          annotation.points.forEach(([px, py], i) => {
            const x = px * displayWidth;
            const y = py * displayHeight;
            if (i === 0) {
              ctx.moveTo(x, y);
            } else {
              ctx.lineTo(x, y);
            }
          });
          ctx.closePath();
          ctx.stroke();
          ctx.fill();

          // Draw label at first point
          const firstPoint = annotation.points[0];
          if (firstPoint) {
            const [px, py] = firstPoint;
            const x = px * displayWidth;
            const y = py * displayHeight;
            const label = `${index + 1}`;
            ctx.fillStyle = color;
            ctx.font = 'bold 14px Arial';
            ctx.fillText(label, x + 4, y + 16);
          }
        }
      });

      setIsLoaded(true);
    };

    img.onerror = () => {
      setError('Failed to load image');
    };

    // Construct full URL from API_BASE_URL
    const fullUrl = imageUrl.startsWith('http')
      ? imageUrl
      : `${API_BASE_URL}${imageUrl}`;

    img.src = fullUrl;
  }, [imageUrl, imageWidth, imageHeight, annotations]);

  if (error) {
    return (
      <div style={{
        padding: '20px',
        textAlign: 'center',
        color: '#dc2626',
        border: '1px solid #fee2e2',
        borderRadius: '6px',
        background: '#fef2f2'
      }}>
        {error}
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        onClick={onImageClick}
        style={{
          cursor: onImageClick ? 'pointer' : 'default',
          border: '1px solid #e5e7eb',
          borderRadius: '6px',
          maxWidth: '100%',
          opacity: isLoaded ? 1 : 0,
          transition: 'opacity 0.3s ease'
        }}
      />
      {!isLoaded && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#6b7280'
        }}>
          Loading image...
        </div>
      )}
    </div>
  );
};

export default AnnotatedTemplateImage;
