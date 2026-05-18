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
  previousChars?: any[];       // old char annotations from recipe — used to inherit conf/text on re-segment
  isSaving: boolean;
  onConfirm: (finalAnnotations: any[]) => void;
  onCancel: () => void;
}

/** Centroid of a char annotation (normalized 0-1 coords). */
function _centerXY(c: any): [number, number] {
  return [(c.x ?? 0) + (c.width ?? 0) / 2, (c.y ?? 0) + (c.height ?? 0) / 2];
}

/** True if center of `c` falls inside the rectangle of `parent`. */
function _isInsideRegion(c: any, parent: any): boolean {
  const [cx, cy] = _centerXY(c);
  return cx >= (parent.x ?? 0)
      && cx <= (parent.x ?? 0) + (parent.width ?? 0)
      && cy >= (parent.y ?? 0)
      && cy <= (parent.y ?? 0) + (parent.height ?? 0);
}

/** Inherit text/conf from previous char into new char if a positional match exists.
 *  - text: prefer OCR-recognized (new char's text). Fallback to previous if OCR empty.
 *  - conf: prefer previous char's value (user-customized). Fallback to new (parent default). */
function _inheritFromPrev(newChar: any, prevChar: any | undefined): any {
  if (!prevChar) return newChar;
  const out = { ...newChar };
  if ((!out.text || out.text === '') && prevChar.text) out.text = prevChar.text;
  if (typeof prevChar.conf === 'number') out.conf = prevChar.conf;
  return out;
}

/** Axis-aligned IoU between two normalized-rect annotations. */
function _iouRect(a: any, b: any): number {
  const ax = a.x ?? 0, ay = a.y ?? 0, aw = a.width ?? 0, ah = a.height ?? 0;
  const bx = b.x ?? 0, by = b.y ?? 0, bw = b.width ?? 0, bh = b.height ?? 0;
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = aw * ah + bw * bh - inter;
  return union > 0 ? inter / union : 0;
}

/** Greedy IoU-based matching of new chars to previous chars.
 *  Returns: new chars (in original order) with conf/text inherited from matched prev.
 *  Unmatched new chars keep their default values (parent conf + OCR text).
 *  Threshold avoids accidental inheritance when boxes barely touch.
 *
 *  Trade-off vs sorted-index match:
 *    - Robust when segmentation drifts, inserts, or deletes chars
 *    - Each prev can match at most one new (no double-inherit)
 *    - O(n*m) pairs, fine for typical char counts (< 50)
 */
function _matchByOverlap(
  newChars: any[],
  prevInRegion: any[],
  iouThreshold: number = 0.3,
): { merged: any[]; inheritedCount: number } {
  if (prevInRegion.length === 0) return { merged: newChars, inheritedCount: 0 };

  type Pair = { newIdx: number; prevIdx: number; iou: number };
  const pairs: Pair[] = [];
  for (let i = 0; i < newChars.length; i++) {
    for (let j = 0; j < prevInRegion.length; j++) {
      const iou = _iouRect(newChars[i], prevInRegion[j]);
      if (iou >= iouThreshold) pairs.push({ newIdx: i, prevIdx: j, iou });
    }
  }
  pairs.sort((a, b) => b.iou - a.iou);

  const newUsed = new Set<number>();
  const prevUsed = new Set<number>();
  const matches = new Map<number, number>(); // newIdx → prevIdx
  for (const p of pairs) {
    if (newUsed.has(p.newIdx) || prevUsed.has(p.prevIdx)) continue;
    matches.set(p.newIdx, p.prevIdx);
    newUsed.add(p.newIdx);
    prevUsed.add(p.prevIdx);
  }

  const merged = newChars.map((nc, i) => {
    const pj = matches.get(i);
    return pj !== undefined ? _inheritFromPrev(nc, prevInRegion[pj]) : nc;
  });
  return { merged, inheritedCount: matches.size };
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
  previousChars,
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
      // Ensure every base annotation has a stable id — otherwise the canvas
      // change handler can't tell "existing" from "newly drawn" and would force
      // them all to type='char'.
      const baseWithIds = baseAnnotations.map((a: any, i: number) =>
        a?.id ? a : { ...a, id: `base-${i}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}` }
      );
      setAnnotations(baseWithIds);
      setAutoSegmenting(true);
      const newChars: any[] = [];
      let totalSegs = 0;
      let failedRegions = 0;

      let inheritedCount = 0;
      for (const ann of baseWithIds) {
        if (ann.shape !== 'rectangle') continue;
        if (ann.type !== 'text' && ann.type !== 'datecode') continue;
        try {
          const res = await recipesAPI.segmentTemplateRegion(
            imageUrl,
            { x: ann.x ?? 0, y: ann.y ?? 0, w: ann.width ?? 0, h: ann.height ?? 0 },
            { withOcr: true },
          );
          if (res.count > 0) {
            // Build raw new chars from segmentation
            const freshChars = res.segments.map((seg) =>
              buildPaddedChar(seg, ann.conf ?? 0.5, imageWidth, imageHeight),
            );
            // Inherit conf/text from previous chars by IoU overlap (greedy match).
            // Unmatched new chars keep default conf and OCR-recognized text.
            const prevInRegion = (previousChars ?? []).filter((c: any) => _isInsideRegion(c, ann));
            const { merged, inheritedCount: matched } = _matchByOverlap(freshChars, prevInRegion);
            inheritedCount += matched;
            for (const c of merged) {
              newChars.push(c);
              totalSegs += 1;
            }
          }
        } catch (e) {
          failedRegions += 1;
          console.error('[TemplateCanvasEditDialog] segment failed for region', ann, e);
        }
      }

      setAnnotations([...baseWithIds, ...newChars]);
      setAutoSegmenting(false);

      if (totalSegs > 0) {
        const inheritMsg = inheritedCount > 0 ? ` — kept text/conf for ${inheritedCount}` : '';
        toast.success(`Auto segmented ${totalSegs} character(s)${inheritMsg} — review and adjust as needed`);
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

      const freshChars = res.segments.map((seg) =>
        buildPaddedChar(seg, ann.conf ?? 0.5, imageWidth, imageHeight),
      );

      // Inherit text/conf from chars CURRENTLY in this region (so users who
      // re-segment can keep their customized values for unchanged chars).
      // Also fall back to recipe's previousChars when the dialog has no chars yet
      // (e.g. user manually deleted them all before re-segmenting).
      const currentInRegion = annotations.filter(
        (a: any) => a.type === 'char' && _isInsideRegion(a, ann),
      );
      const prevSource = currentInRegion.length > 0
        ? currentInRegion
        : (previousChars ?? []).filter((c: any) => _isInsideRegion(c, ann));
      const { merged: newChars, inheritedCount } = _matchByOverlap(freshChars, prevSource);

      // Replace any old chars in this region with the new chars (insert right
      // after the region annotation for nicer ordering).
      setAnnotations((prev) => {
        const filtered = prev.filter(
          (a: any) => !(a.type === 'char' && _isInsideRegion(a, ann)),
        );
        const regionIdx = filtered.findIndex((a: any) => a === ann);
        const insertAt = regionIdx >= 0 ? regionIdx + 1 : filtered.length;
        return [...filtered.slice(0, insertAt), ...newChars, ...filtered.slice(insertAt)];
      });

      const ocrPreview = res.full_text ? ` ("${res.full_text}")` : '';
      const inheritMsg = inheritedCount > 0 ? ` — kept text/conf for ${inheritedCount}` : '';
      toast.success(`Re-segmented ${res.count} char(s)${ocrPreview}${inheritMsg}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Segmentation failed');
    } finally {
      setSegmentingIdx(null);
    }
  };

  // LOCKED in this dialog: type and conf are immutable for ALL annotations.
  // Non-char regions are also fully locked (no move/resize/delete). Only char
  // annotations can be moved/resized/deleted/redrawn; their text is editable.
  const handleAnnotationTypeChange = (_index: number, _type: string) => {
    toast.info('Type is locked in this dialog');
  };
  const handleAnnotationTextChange = (index: number, text: string) => {
    setAnnotations((prev) => prev.map((a, i) => (i === index ? { ...a, text } : a)));
  };
  const handleAnnotationConfChange = (_index: number, _conf: number) => {
    toast.info('Confidence is locked — inherited from previous template');
  };
  const handleDeleteAnnotation = (index: number) => {
    const ann = annotations[index];
    if (ann && ann.type !== 'char') {
      toast.info('Only character annotations can be deleted here');
      return;
    }
    setAnnotations((prev) => prev.filter((_, i) => i !== index));
    setSelectedAnnotation(null);
  };

  /** Canvas drag/resize on a non-char region must be reverted to keep geometry locked.
   *  Compare each incoming annotation against the current state by `id`; revert geometry
   *  + type + conf for non-char items. For chars, only force type='char' and lock conf. */
  const handleAnnotationsChange = (next: any[]) => {
    const byId = new Map<string, any>();
    for (const a of annotations) if (a?.id) byId.set(a.id, a);
    // The previous-state size: anything beyond this index in `next` was added
    // by the canvas (Rectangle tool). We rely on TemplateEditor preserving order.
    const prevLen = annotations.length;
    const merged = next.map((newAnn, idx) => {
      // Primary lookup: by id
      let prev = newAnn?.id ? byId.get(newAnn.id) : undefined;
      // Fallback: positional lookup (same index in old list) — handles the case
      // where an existing annotation lost its id somewhere in the pipeline.
      if (!prev && idx < prevLen) prev = annotations[idx];

      if (!prev) {
        // Truly new annotation (index >= prevLen AND no id match) — force char.
        return { ...newAnn, type: 'char' };
      }
      if (prev.type !== 'char') {
        return {
          ...newAnn,
          x: prev.x, y: prev.y,
          width: prev.width, height: prev.height,
          type: prev.type,
          conf: prev.conf,
        };
      }
      return { ...newAnn, type: 'char', conf: prev.conf };
    });
    setAnnotations(merged);
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
              onAnnotationsChange={handleAnnotationsChange}
              selectedAnnotation={selectedAnnotation}
              onSelectAnnotation={setSelectedAnnotation}
              fabricCanvasRef={fabricCanvasRef as any}
              // Lock non-char regions, hide polygon tool, and force Rectangle to draw chars.
              lockedTypes={['text', 'datecode', 'template', 'crop_area']}
              disableDrawing                // hides Polygon button only
              defaultDrawType="char"       // Rectangle draws char annotations
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
              readOnlyType                      // lock type select for all
              readOnlyConf                      // lock conf input for all
              canDelete={(a: any) => a.type === 'char'}  // only chars deletable
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
