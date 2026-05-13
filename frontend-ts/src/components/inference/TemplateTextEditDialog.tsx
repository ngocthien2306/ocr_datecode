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

  useEffect(() => {
    if (isOpen) {
      setEditedItems(items.map(item => ({ ...item })));
    }
  }, [isOpen, items]);

  if (!isOpen) return null;

  const handleTextChange = (idx: number, newText: string) => {
    setEditedItems(prev => prev.map(item => item.idx === idx ? { ...item, newText } : item));
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
          <div className="tted-col-header">
            <span>Region</span>
            <span>Old text</span>
            <span />
            <span>New text</span>
          </div>

          {editedItems.map(item => (
            <div key={item.idx} className="tted-row">
              <div className="tted-label">
                <span className={`tted-type-badge tted-type-${item.type}`}>{item.type}</span>
                <span className="tted-label-text">{item.label}</span>
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
                placeholder="New text..."
                spellCheck={false}
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
