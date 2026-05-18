import { useEffect, useRef, useState } from 'react';
import type { Canvas as FabricCanvas } from 'fabric';
import TemplateEditor from '@/components/recipe/TemplateEditorRefactored';
import AnnotationsPanel from '@/components/shared/AnnotationsPanel';
import { recipesAPI } from '@/services/recipes';
import { API_BASE_URL } from '@/config/api';
import { useToast } from '@/contexts/ToastContext';
import '@/styles/TemplateCanvasEditDialog.css';

interface Props {
  isOpen: boolean;
  imageUrl: string;            // server-relative path (e.g. /api/recipes/templates/images/xxx.jpg)
  imageWidth: number;
  imageHeight: number;
  templateName?: string;
  baseAnnotations: any[];      // non-char annotations with coords already updated from new frame
  isSaving: boolean;
  onConfirm: (finalAnnotations: any[]) => void;
  onCancel: () => void;
}

// Pad chars by +4px on width AND +4px on height (2px each side), in image pixel units.
const CHAR_PAD_PX = 4;

function buildPaddedChar(
  seg: { x: number; y: number; w: number; h: number; expected_text?: string | null },
  parentConf: number,
  imageWidth: number,
  imageHeight: number,
): any {
  const halfDxN = (CHAR_PAD_PX / 2) / imageWidth;
  const halfDyN = (CHAR_PAD_PX / 2) / imageHeight;
  const dxN = CHAR_PAD_PX / imageWidth;
  const dyN = CHAR_PAD_PX / imageHeight;

  let newX = seg.x - halfDxN;
  let newY = seg.y - halfDyN;
  let newW = seg.w + dxN;
  let newH = seg.h + dyN;

  // Clip into [0,1]
  if (newX < 0) { newW += newX; newX = 0; }
  if (newY < 0) { newH += newY; newY = 0; }
  if (newX + newW > 1) newW = 1 - newX;
  if (newY + newH > 1) newH = 1 - newY;

  return {
    id: `annotation-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    type: 'char',
    shape: 'rectangle',
    x: newX,
    y: newY,
    width: newW,
    height: newH,
    text: (seg.expected_text ?? '').toString(),
    conf: parentConf,
  };
}

export default function TemplateCanvasEditDialog({
  isOpen,
  imageUrl,
  imageWidth,
  imageHeight,
  templateName,
  baseAnnotations,
  isSaving,
  onConfirm,
  onCancel,
}: Props) {
  const [annotations, setAnnotations] = useState<any[]>([]);
  const [selectedAnnotation, setSelectedAnnotation] = useState<number | null>(null);
  const [autoSegmenting, setAutoSegmenting] = useState(false);
  const [segmentingIdx, setSegmentingIdx] = useState<number | null>(null);
  const fabricCanvasRef = useRef<FabricCanvas | null>(null);
  const toast = useToast();
  const hasRunInitialRef = useRef(false);

  const fullImageUrl = imageUrl.startsWith('http') ? imageUrl : `${API_BASE_URL}${imageUrl}`;

  // On open: load base annotations and auto-segment every text/datecode region into fresh chars.
  useEffect(() => {
    if (!isOpen) {
      hasRunInitialRef.current = false;
      setAnnotations([]);
      setSelectedAnnotation(null);
      return;
    }
    if (hasRunInitialRef.current) return;
    hasRunInitialRef.current = true;

    const run = async () => {
      setAnnotations(baseAnnotations);
      setAutoSegmenting(true);
      const newChars: any[] = [];
      let totalSegs = 0;
      let failedRegions = 0;

      for (const ann of baseAnnotations) {
        if (ann.shape !== 'rectangle') continue;
        if (ann.type !== 'text' && ann.type !== 'datecode') continue;
        try {
          const res = await recipesAPI.segmentTemplateRegion(
            imageUrl,
            { x: ann.x ?? 0, y: ann.y ?? 0, w: ann.width ?? 0, h: ann.height ?? 0 },
            { withOcr: true },
          );
          if (res.count > 0) {
            for (const seg of res.segments) {
              newChars.push(buildPaddedChar(seg, ann.conf ?? 0.5, imageWidth, imageHeight));
              totalSegs += 1;
            }
          }
        } catch (e) {
          failedRegions += 1;
          console.error('[TemplateCanvasEditDialog] segment failed for region', ann, e);
        }
      }

      setAnnotations([...baseAnnotations, ...newChars]);
      setAutoSegmenting(false);

      if (totalSegs > 0) {
        toast.success(`Auto segmented ${totalSegs} character(s) — review and adjust as needed`);
      } else if (failedRegions > 0) {
        toast.warning('Auto segmentation failed for all regions — draw chars manually');
      } else {
        toast.info('No text/datecode regions to segment');
      }
    };
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  // Manual re-segment a single text/datecode (replaces only chars whose parent is this region).
  const handleAutoSegmentSingle = async (index: number) => {
    const ann = annotations[index];
    if (!ann || ann.shape !== 'rectangle') return;
    if (ann.type !== 'text' && ann.type !== 'datecode') {
      toast.warning('Segment is only supported on Text OCR or Date Code regions');
      return;
    }

    setSegmentingIdx(index);
    try {
      const res = await recipesAPI.segmentTemplateRegion(
        imageUrl,
        { x: ann.x ?? 0, y: ann.y ?? 0, w: ann.width ?? 0, h: ann.height ?? 0 },
        { withOcr: true },
      );
      if (res.count === 0) {
        toast.warning('No characters found in this region');
        return;
      }
      const newChars = res.segments.map((seg) =>
        buildPaddedChar(seg, ann.conf ?? 0.5, imageWidth, imageHeight),
      );
      // Insert right after the region for nicer ordering.
      setAnnotations((prev) => {
        const updated = [...prev];
        updated.splice(index + 1, 0, ...newChars);
        return updated;
      });
      const ocrPreview = res.full_text ? ` ("${res.full_text}")` : '';
      toast.success(`Created ${res.count} char annotation(s)${ocrPreview}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Segmentation failed');
    } finally {
      setSegmentingIdx(null);
    }
  };

  const handleAnnotationTypeChange = (index: number, type: string) => {
    setAnnotations((prev) => prev.map((a, i) => (i === index ? { ...a, type } : a)));
  };
  const handleAnnotationTextChange = (index: number, text: string) => {
    setAnnotations((prev) => prev.map((a, i) => (i === index ? { ...a, text } : a)));
  };
  const handleAnnotationConfChange = (index: number, conf: number) => {
    setAnnotations((prev) => prev.map((a, i) => (i === index ? { ...a, conf } : a)));
  };
  const handleDeleteAnnotation = (index: number) => {
    setAnnotations((prev) => prev.filter((_, i) => i !== index));
    setSelectedAnnotation(null);
  };

  if (!isOpen) return null;

  const charCount = annotations.filter((a) => a.type === 'char').length;
  const regionCount = annotations.filter((a) => a.type === 'text' || a.type === 'datecode').length;

  return (
    <div className="tced-overlay">
      <div className="tced-dialog">
        <div className="tced-header">
          <div>
            <h3>Edit Template Annotations</h3>
            {templateName && <span className="tced-subtitle">{templateName}</span>}
          </div>
          <button className="tced-close" onClick={onCancel} disabled={isSaving} aria-label="Close">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        <div className="tced-body">
          <div className="tced-canvas-pane">
            {autoSegmenting && (
              <div className="tced-overlay-loading">
                <span className="tced-spin" />
                Auto-segmenting characters…
              </div>
            )}
            <TemplateEditor
              templateImage={fullImageUrl}
              annotations={annotations as any}
              onAnnotationsChange={(next: any[]) => setAnnotations(next)}
              selectedAnnotation={selectedAnnotation}
              onSelectAnnotation={setSelectedAnnotation}
              fabricCanvasRef={fabricCanvasRef as any}
            />
          </div>

          <div className="tced-sidebar-pane">
            <AnnotationsPanel
              annotations={annotations as any}
              selectedAnnotation={selectedAnnotation}
              onSelectAnnotation={setSelectedAnnotation}
              onAnnotationTypeChange={handleAnnotationTypeChange}
              onAnnotationTextChange={handleAnnotationTextChange}
              onAnnotationConfChange={handleAnnotationConfChange}
              onDeleteAnnotation={handleDeleteAnnotation}
              onAutoSegment={handleAutoSegmentSingle}
              segmenting={segmentingIdx !== null}
              fabricCanvasRef={fabricCanvasRef as any}
              imageWidth={imageWidth}
              imageHeight={imageHeight}
            />
          </div>
        </div>

        <div className="tced-footer">
          <div className="tced-footer-info">
            {regionCount} region(s) • {charCount} char(s)
          </div>
          <div className="tced-footer-actions">
            <button className="tced-btn-cancel" onClick={onCancel} disabled={isSaving}>
              Cancel
            </button>
            <button
              className="tced-btn-confirm"
              onClick={() => onConfirm(annotations)}
              disabled={isSaving || autoSegmenting}
            >
              {isSaving ? 'Saving…' : 'Save Template'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
