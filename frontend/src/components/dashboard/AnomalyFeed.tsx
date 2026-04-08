'use client';

import { useEffect, useRef, useState } from 'react';
import { getLatestAnomalies, type AnomalyRecord } from '@/lib/api';
import { POLL_INTERVAL_MS, SEVERITY_COLORS } from '@/lib/constants';
import { Badge } from '@/components/ui/Badge';
import { Card, CardSkeleton } from '@/components/ui/Card';
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
          className={`h-full rounded-full transition-all duration-300 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-8 text-right">{pct}%</span>
    </div>
  );
}

function AnomalyItem({
  anomaly,
  isNew,
  onClick,
}: {
  anomaly: AnomalyRecord;
  isNew: boolean;
  onClick: (a: AnomalyRecord) => void;
}) {
  const sev = anomaly.severity as 'critical' | 'high' | 'medium' | 'low';
  const colors = SEVERITY_COLORS[sev] ?? SEVERITY_COLORS.low;

  return (
    <button
      onClick={() => onClick(anomaly)}
      className={`
        w-full text-left p-3 rounded-lg border transition-all duration-200 cursor-pointer
        hover:border-gray-500 hover:bg-gray-750
        ${isNew ? 'animate-slide-in' : ''}
        ${anomaly.severity === 'critical' ? 'border-red-800/60 bg-red-900/10' :
          anomaly.severity === 'high' ? 'border-orange-800/60 bg-orange-900/10' :
          'border-gray-700 bg-gray-800/50'}
      `}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Badge variant={sev}>{anomaly.severity}</Badge>
          <span className="text-sm font-medium text-gray-200">{anomaly.host_id}</span>
        </div>
        <span className="text-xs text-gray-500">
          {formatRelativeTime(anomaly.detected_at)}
        </span>
      </div>

      <ScoreBar score={anomaly.combined_score} />

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <span className="text-xs text-gray-400">
          Worst:{' '}
          <span className={`font-medium ${colors.text}`}>{anomaly.worst_feature}</span>
        </span>
        <span className="text-xs text-gray-500">
          via{' '}
          <span className="text-gray-400 font-mono">{anomaly.detection_method}</span>
        </span>
      </div>
    </button>
  );
}

interface AnomalyFeedProps {
  /** Called when the user clicks an anomaly to trigger AI analysis. */
  onSelectAnomaly: (anomaly: AnomalyRecord) => void;
}

/**
 * Live-scrolling feed of detected anomalies.
 *
 * - Auto-refreshes every 5 s via /anomalies/latest.
 * - New items slide in with a CSS animation.
 * - Click an item to send it to the RCA panel.
 * - Empty state shows a "system healthy" message.
 */
export function AnomalyFeed({ onSelectAnomaly }: AnomalyFeedProps) {
  const [anomalies, setAnomalies] = useState<AnomalyRecord[]>([]);
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const prevIdsRef = useRef<Set<string>>(new Set());
  const mountedRef = useRef(true);

  async function fetchAnomalies() {
    try {
      const data = await getLatestAnomalies(undefined, undefined, 30);
      if (!mountedRef.current) return;

      const incoming = data.anomalies;
      const incomingIds = new Set(incoming.map((a) => a.record_id));
      const fresh = new Set<string>();
      incomingIds.forEach((id) => {
        if (!prevIdsRef.current.has(id)) fresh.add(id);
      });

      prevIdsRef.current = incomingIds;
      setAnomalies(incoming);
      setNewIds(fresh);
      setError(null);

      // Clear new-item highlight after animation completes
      if (fresh.size > 0) {
        setTimeout(() => {
          if (mountedRef.current) setNewIds(new Set());
        }, 800);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load anomalies');
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }

  useEffect(() => {
    mountedRef.current = true;
    fetchAnomalies();
    const id = setInterval(fetchAnomalies, POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const headerAction = anomalies.length > 0 && (
    <span className="text-xs text-gray-500">{anomalies.length} events</span>
  );

  if (loading) return <CardSkeleton className="h-72" />;

  return (
    <Card
      title="Anomaly Feed"
      subtitle="Click an event to analyze with AI"
      headerAction={headerAction}
      className="h-full flex flex-col"
    >
      {error && <AlertBanner level="error" message={error} className="mb-3" />}

      <div className="flex-1 overflow-y-auto space-y-2 max-h-72 pr-1">
        {anomalies.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-44 text-center">
            <span className="text-2xl mb-2">✓</span>
            <p className="text-gray-400 text-sm font-medium">
              No anomalies detected
            </p>
            <p className="text-gray-600 text-xs mt-1">System healthy</p>
          </div>
        ) : (
          anomalies.map((a) => (
            <AnomalyItem
              key={a.record_id}
              anomaly={a}
              isNew={newIds.has(a.record_id)}
              onClick={onSelectAnomaly}
            />
          ))
        )}
      </div>
    </Card>
  );
}
