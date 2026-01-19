/**
 * Utility functions for AI Agent
 */

import { SuggestedAction } from '../types/agent';

/**
 * Parse suggested actions from agent response
 *
 * Detects patterns like:
 * - "Bạn muốn:\n1. Option 1\n2. Option 2"
 * - "1. 📊 Thống kê\n2. 🔧 Services"
 * - "• Option A\n• Option B"
 */
export function parseSuggestedActions(content: string): SuggestedAction[] {
  const actions: SuggestedAction[] = [];

  // Pattern 1: Numbered list (1. 2. 3.)
  const numberedPattern = /^\d+\.\s*(.+)$/gm;
  const numberedMatches = content.match(numberedPattern);

  if (numberedMatches && numberedMatches.length >= 2) {
    numberedMatches.forEach(match => {
      const label = match.replace(/^\d+\.\s*/, '').trim();
      // Remove emojis for message, keep for label
      const message = label.replace(/[\u{1F300}-\u{1F9FF}]/gu, '').trim();

      actions.push({
        label,
        message
      });
    });
    return actions;
  }

  // Pattern 2: Bullet points (• or -)
  const bulletPattern = /^[•\-]\s*(.+)$/gm;
  const bulletMatches = content.match(bulletPattern);

  if (bulletMatches && bulletMatches.length >= 2) {
    bulletMatches.forEach(match => {
      const label = match.replace(/^[•\-]\s*/, '').trim();
      const message = label.replace(/[\u{1F300}-\u{1F9FF}]/gu, '').trim();

      actions.push({
        label,
        message
      });
    });
    return actions;
  }

  return actions;
}

/**
 * Check if message content contains suggested actions
 */
export function hasSuggestedActions(content: string): boolean {
  // Check for numbered list with at least 2 items
  const numberedPattern = /^\d+\.\s*(.+)$/gm;
  const numberedMatches = content.match(numberedPattern);
  if (numberedMatches && numberedMatches.length >= 2) {
    return true;
  }

  // Check for bullet points with at least 2 items
  const bulletPattern = /^[•\-]\s*(.+)$/gm;
  const bulletMatches = content.match(bulletPattern);
  if (bulletMatches && bulletMatches.length >= 2) {
    return true;
  }

  return false;
}
