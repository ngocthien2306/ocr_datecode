import { useState, useEffect } from 'react';
import '@/styles/TemplateTextEditDialog.css';

export interface TextEditItem {
  idx: number;
  type: string;
  label: string;
  oldText: string;
  newText: string;
}

interface Props {
  isOpen: boolean;
  templateName?: string;
  items: TextEditItem[];
  isLoading: boolean;
  onConfirm: (items: TextEditItem[]) => void;
  onCancel: () => void;
}

export default function TemplateTextEditDialog({ isOpen, templateName, items, isLoading, onConfirm, onCancel }: Props) {
  const [editedItems, setEditedItems] = useState<TextEditItem[]>([]);
  const [draggedIdx, setDraggedIdx] = useState<number | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    if (isOpen) {
      setEditedItems(items.map(item => ({ ...item })));
      setDraggedIdx(null);
      setHoverIdx(null);
    }
  }, [isOpen, items]);

  if (!isOpen) return null;

  const handleTextChange = (idx: number, newText: string) => {
    setEditedItems(prev => prev.map(item => item.idx === idx ? { ...item, newText } : item));
  };

  const handleDragStart = (itemIdx: number) => (e: React.DragEvent) => {
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(itemIdx));
    setDraggedIdx(itemIdx);
  };

  const handleDragOver = (itemIdx: number) => (e: React.DragEvent) => {
    if (draggedIdx === null) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (hoverIdx !== itemIdx) setHoverIdx(itemIdx);
  };

  const handleDrop = (targetItemIdx: number) => (e: React.DragEvent) => {
    e.preventDefault();
    if (draggedIdx === null || draggedIdx === targetItemIdx) {
      setDraggedIdx(null);
      setHoverIdx(null);
      return;
    }
    setEditedItems(prev => {
      const fromPos = prev.findIndex(i => i.idx === draggedIdx);
      const toPos = prev.findIndex(i => i.idx === targetItemIdx);
      if (fromPos === -1 || toPos === -1) return prev;
      const next = [...prev];
      const moved = next.splice(fromPos, 1)[0];
      if (!moved) return prev;
      next.splice(toPos, 0, moved);
      return next;
    });
    setDraggedIdx(null);
    setHoverIdx(null);
  };

  const handleDragEnd = () => {
    setDraggedIdx(null);
    setHoverIdx(null);
  };

  return (
    <div className="tted-overlay">
      <div className="tted-dialog">
        <div className="tted-header">
          <div>
            <h3>Review Template Text</h3>
            {templateName && <span className="tted-subtitle">{templateName}</span>}
          </div>
          <button className="tted-close" onClick={onCancel} disabled={isLoading}>✕</button>
        </div>

        <div className="tted-body">
          {editedItems.map(item => (
            <div
              key={item.idx}
              className={`tted-row tted-row-${item.type}${draggedIdx === item.idx ? ' is-dragging' : ''}${hoverIdx === item.idx && draggedIdx !== null && draggedIdx !== item.idx ? ' is-drop-target' : ''}`}
              draggable={item.type === 'char'}
              onDragStart={item.type === 'char' ? handleDragStart(item.idx) : undefined}
              onDragOver={handleDragOver(item.idx)}
              onDrop={handleDrop(item.idx)}
              onDragEnd={handleDragEnd}
            >
              <div className="tted-label">
                <span className={`tted-type-badge tted-type-${item.type}`}>{item.type}</span>
                {item.type !== 'char' && <span className="tted-label-text">{item.label}</span>}
              </div>

              <div className="tted-old-text">
                {item.oldText || <em>empty</em>}
              </div>

              <div className="tted-arrow">→</div>

              <input
                className="tted-input"
                value={item.newText}
                onChange={e => handleTextChange(item.idx, e.target.value)}
                disabled={isLoading}
                placeholder={item.type === 'char' ? '' : 'New text...'}
                spellCheck={false}
                maxLength={item.type === 'char' ? 3 : undefined}
              />
            </div>
          ))}
        </div>

        <div className="tted-footer">
          <button className="tted-btn-cancel" onClick={onCancel} disabled={isLoading}>
            Cancel
          </button>
          <button className="tted-btn-confirm" onClick={() => onConfirm(editedItems)} disabled={isLoading}>
            {isLoading ? 'Saving...' : 'Confirm & Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
