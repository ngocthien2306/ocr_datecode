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

interface ImageZoomModalProps {
  isOpen: boolean;
  onClose: () => void;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  annotations: Annotation[];
}

const ImageZoomModal: React.FC<ImageZoomModalProps> = ({
  isOpen,
  onClose,
  imageUrl,
  imageWidth,
  imageHeight,
  annotations
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  const getColorForType = (type: string): string => {
    const config = TYPE_CONFIGS.find(c => c.value === type);
    return config?.color || '#ffffff';
  };

  useEffect(() => {
    if (!isOpen) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      // Calculate display size (max 90% of viewport, maintain aspect ratio)
      const maxWidth = window.innerWidth * 0.85;
      const maxHeight = window.innerHeight * 0.85;
      const scale = Math.min(maxWidth / imageWidth, maxHeight / imageHeight, 1.5);
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
        ctx.lineWidth = 3;
        ctx.fillStyle = color + '33'; // 20% opacity for fill

        if (annotation.shape === 'rectangle' && annotation.x !== undefined) {
          const x = annotation.x * displayWidth;
          const y = annotation.y! * displayHeight;
          const width = annotation.width! * displayWidth;
          const height = annotation.height! * displayHeight;

          ctx.strokeRect(x, y, width, height);
          ctx.fillRect(x, y, width, height);

          // Draw label
          const label = `${index + 1}`;
          ctx.fillStyle = color;
          ctx.font = 'bold 18px Arial';
          ctx.fillText(label, x + 6, y + 22);
        } else if (annotation.shape === 'polygon' && annotation.points && annotation.points.length > 0) {
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

          const firstPoint = annotation.points[0];
          if (firstPoint) {
            const [px, py] = firstPoint;
            const x = px * displayWidth;
            const y = py * displayHeight;
            const label = `${index + 1}`;
            ctx.fillStyle = color;
            ctx.font = 'bold 18px Arial';
            ctx.fillText(label, x + 6, y + 22);
          }
        }
      });

      setIsLoaded(true);
    };

    img.onerror = () => {
      console.error('Failed to load image');
    };

    const fullUrl = imageUrl.startsWith('http')
      ? imageUrl
      : `${API_BASE_URL}${imageUrl}`;

    img.src = fullUrl;
  }, [isOpen, imageUrl, imageWidth, imageHeight, annotations]);

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2000,
        padding: '20px'
      }}
      onClick={onClose}
    >
      <div
        style={{
          position: 'relative',
          maxWidth: '90vw',
          maxHeight: '90vh'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '-40px',
            right: '0',
            background: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '32px',
            height: '32px',
            cursor: 'pointer',
            fontSize: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2001
          }}
        >
          ×
        </button>
        <canvas
          ref={canvasRef}
          style={{
            border: '2px solid white',
            borderRadius: '8px',
            maxWidth: '100%',
            maxHeight: '85vh',
            opacity: isLoaded ? 1 : 0,
            transition: 'opacity 0.3s ease'
          }}
        />
        {!isLoaded && (
          <div style={{
            color: 'white',
            fontSize: '18px',
            textAlign: 'center'
          }}>
            Loading...
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageZoomModal;
