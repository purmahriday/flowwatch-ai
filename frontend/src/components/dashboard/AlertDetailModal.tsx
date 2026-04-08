'use client';

import { useEffect, useRef, useState } from 'react';
import { getLatestAnomalies, type AlertRecord, type AnomalyRecord } from '@/lib/api';
import { SEVERITY_COLORS } from '@/lib/constants';
import { Badge } from '@/components/ui/Badge';
import { AlertBanner } from '@/components/ui/Alert';

function formatRelativeTime(isoTs: string): string {
  const diffMs = Date.now() - new Date(isoTs).getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} min${diffMin !== 1 ? 's' : ''} ago`;
  const diffHr = Math.floor(diffMin / 60);
  return `${diffHr} hr${diffHr !== 1 ? 's' : ''} ago`;
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    score > 0.8 ? 'bg-red-500' :
    score > 0.6 ? 'bg-orange-500' :
    score > 0.4 ? 'bg-yellow-500' :
    'bg-blue-500';

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-8 text-right tabular-nums">{pct}%</span>
    </div>
  );
}

function AnomalyRow({ anomaly }: { anomaly: AnomalyRecord }) {
  const sev = anomaly.severity as 'critical' | 'high' | 'medium' | 'low';

  return (
    <div className="p-3 rounded-lg bg-gray-800/60 border border-gray-700 space-y-2">
      {/* Timestamp + method */}
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-gray-400">
          {new Date(anomaly.detected_at).toLocaleTimeString()} — {formatRelativeTime(anomaly.detected_at)}
        </span>
        <span className="text-xs font-mono bg-gray-700 text-gray-300 px-2 py-0.5 rounded">
          {anomaly.detection_method}
        </span>
      </div>

      {/* Scores row */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-400">
        <span>
          LSTM: <span className="text-gray-200 font-mono">{anomaly.lstm_result.anomaly_score.toFixed(3)}</span>
        </span>
        <span>
          IF: <span className="text-gray-200 font-mono">{anomaly.if_result.anomaly_score.toFixed(3)}</span>
        </span>
        <span>
          Worst: <span className={`font-medium ${SEVERITY_COLORS[sev]?.text ?? 'text-gray-300'}`}>
            {anomaly.worst_feature}
          </span>
        </span>
      </div>

      {/* Combined score bar */}
      <ScoreBar score={anomaly.combined_score} />
    </div>
  );
}

interface AlertDetailModalProps {
  alert: AlertRecord | null;
  onClose: () => void;
  onAnalyzeWithAI: (anomaly: AnomalyRecord) => void;
}

/**
 * Full-screen overlay modal showing alert details and host anomaly history.
 *
 * - Alert details shown at top: host, severity, timestamp, score.
 * - Fetches up to 50 anomalies for the alert's host from /anomalies/latest.
 * - Each anomaly row shows timestamp, LSTM/IF scores, combined score bar, worst feature, method.
 * - "Analyze with AI" picks the most recent anomaly and hands it to RCAPanel.
 * - Close button in top-right corner.
 */
export function AlertDetailModal({ alert, onClose, onAnalyzeWithAI }: AlertDetailModalProps) {
  const [anomalies, setAnomalies] = useState<AnomalyRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    if (!alert) return;

    setAnomalies([]);
    setError(null);
    setLoading(true);

    getLatestAnomalies(alert.host_id, undefined, 50)
      .then((data) => {
        if (mountedRef.current) setAnomalies(data.anomalies);
      })
      .catch((err) => {
        if (mountedRef.current) {
          setError(err instanceof Error ? err.message : 'Failed to load anomalies');
        }
      })
      .finally(() => {
        if (mountedRef.current) setLoading(false);
      });

    return () => { mountedRef.current = false; };
  }, [alert?.alert_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  if (!alert) return null;

  const sev = alert.severity as 'critical' | 'high' | 'medium' | 'low';

  function handleAnalyzeWithAI() {
    const target = anomalies[0] ?? null;
    if (target) {
      onClose();
      onAnalyzeWithAI(target);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative w-full max-w-2xl bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl flex flex-col max-h-[90vh]">

        {/* ── Header ── */}
        <div className="flex items-start justify-between gap-4 px-6 py-4 border-b border-gray-800">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge variant={sev}>{alert.severity}</Badge>
              {alert.acknowledged && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-gray-500/20 text-gray-400 border border-gray-500/30 uppercase tracking-wide">
                  ACK
                </span>
              )}
              <span className="text-base font-semibold text-gray-100">{alert.host_id}</span>
            </div>
            <p className="text-xs text-gray-500">
              {new Date(alert.timestamp).toLocaleString()} · Score: {alert.combined_score.toFixed(3)} · {alert.worst_feature}
            </p>
            <p className="text-xs text-gray-600">{alert.message}</p>
          </div>
          <button
            onClick={onClose}
            className="shrink-0 w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-100 hover:bg-gray-700 transition-colors"
            aria-label="Close modal"
          >
            ✕
          </button>
        </div>

        {/* ── Anomaly list ── */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-2">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
            Anomaly History — {alert.host_id}
          </p>

          {error && <AlertBanner level="error" message={error} className="mb-3" />}

          {loading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-20 bg-gray-800 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : anomalies.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-center">
              <p className="text-gray-500 text-sm">No anomaly records for {alert.host_id}</p>
            </div>
          ) : (
            anomalies.map((a) => (
              <AnomalyRow key={a.record_id} anomaly={a} />
            ))
          )}
        </div>

        {/* ── Footer ── */}
        <div className="flex gap-3 px-6 py-4 border-t border-gray-800">
          <button
            onClick={handleAnalyzeWithAI}
            disabled={anomalies.length === 0 || loading}
            className="flex-1 px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Analyze with AI
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
