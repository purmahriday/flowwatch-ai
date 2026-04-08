'use client';

import { useEffect, useRef, useState } from 'react';
import {
  getRecentAlerts,
  getRecentTelemetry,
  acknowledgeAlert,
  type AlertRecord,
  type TelemetryRecord,
} from '@/lib/api';
import { POLL_INTERVAL_MS, SEVERITY_COLORS } from '@/lib/constants';
import { Badge } from '@/components/ui/Badge';
import { Card, CardSkeleton } from '@/components/ui/Card';
import { AlertBanner } from '@/components/ui/Alert';

// ─── Thresholds ────────────────────────────────────────────────────────────────

const THRESHOLDS = {
  latency_ms: 200,
  packet_loss_pct: 5,
  dns_failure_rate: 0.1,
  jitter_ms: 50,
};

// ─── Peak telemetry per host ───────────────────────────────────────────────────

interface HostPeaks {
  latency_ms: number;
  packet_loss_pct: number;
  dns_failure_rate: number;
  jitter_ms: number;
}

function computePeaks(records: TelemetryRecord[]): HostPeaks {
  let latency = 0, loss = 0, dns = 0, jitter = 0;
  for (const r of records) {
    if (r.latency_ms > latency) latency = r.latency_ms;
    if (r.packet_loss_pct > loss) loss = r.packet_loss_pct;
    if (r.dns_failure_rate > dns) dns = r.dns_failure_rate;
    if (r.jitter_ms > jitter) jitter = r.jitter_ms;
  }
  return { latency_ms: latency, packet_loss_pct: loss, dns_failure_rate: dns, jitter_ms: jitter };
}

// ─── Anomaly type label ────────────────────────────────────────────────────────

function getAnomalyLabel(peaks: HostPeaks | null): string {
  if (!peaks) return '';
  const lat = peaks.latency_ms > THRESHOLDS.latency_ms;
  const loss = peaks.packet_loss_pct > THRESHOLDS.packet_loss_pct;
  const dns = peaks.dns_failure_rate > THRESHOLDS.dns_failure_rate;
  const jit = peaks.jitter_ms > THRESHOLDS.jitter_ms;

  if (lat && loss && dns && jit) return '🚨 CASCADE — Full Outage';
  if (lat && loss) return '🔥 Network Congestion';
  if (lat) return '⚡ Latency Spike';
  if (loss) return '📦 Packet Loss';
  if (dns) return '🌐 DNS Failure';
  if (jit) return '〰️ High Jitter';
  return '';
}

// ─── Metric row ────────────────────────────────────────────────────────────────

function MetricRow({
  label,
  display,
  bad,
}: {
  label: string;
  display: string;
  bad: boolean | null;
}) {
  return (
    <div className="flex items-center gap-2 py-0.5">
      <span className="text-xs text-gray-500 w-20 shrink-0">{label}</span>
      <span className={`text-xs font-mono flex-1 ${bad === null ? 'text-gray-600' : bad ? 'text-red-400' : 'text-gray-300'}`}>
        {display}
      </span>
      <span className={`text-xs font-bold shrink-0 ${bad === null ? 'text-gray-700' : bad ? 'text-red-400' : 'text-green-400'}`}>
        {bad === null ? '—' : bad ? '↑ BAD' : '✓'}
      </span>
    </div>
  );
}

function formatRelativeTime(isoTs: string): string {
  const diffMs = Date.now() - new Date(isoTs).getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  return `${Math.floor(diffMin / 60)}h ago`;
}

// ─── Alert card ────────────────────────────────────────────────────────────────

function AlertItem({
  alert,
  isNew,
  peaks,
  peaksLoading,
  onAnalyze,
  onAcknowledge,
}: {
  alert: AlertRecord;
  isNew: boolean;
  peaks: HostPeaks | null;
  peaksLoading: boolean;
  onAnalyze: (a: AlertRecord) => void;
  onAcknowledge: (alertId: string) => void;
}) {
  const [acking, setAcking] = useState(false);
  const sev = alert.severity as 'critical' | 'high' | 'medium' | 'low';

  async function handleAcknowledge(e: React.MouseEvent) {
    e.stopPropagation();
    if (alert.acknowledged || acking) return;
    setAcking(true);
    try {
      await acknowledgeAlert(alert.alert_id);
      onAcknowledge(alert.alert_id);
    } finally {
      setAcking(false);
    }
  }

  const anomalyLabel = peaks ? getAnomalyLabel(peaks) : '';

  // Determine per-metric bad status
  const latBad = peaks !== null ? peaks.latency_ms > THRESHOLDS.latency_ms : null;
  const lossBad = peaks !== null ? peaks.packet_loss_pct > THRESHOLDS.packet_loss_pct : null;
  const dnsBad = peaks !== null ? peaks.dns_failure_rate > THRESHOLDS.dns_failure_rate : null;
  const jitBad = peaks !== null ? peaks.jitter_ms > THRESHOLDS.jitter_ms : null;

  const latDisplay = peaks !== null ? `${Math.round(peaks.latency_ms)}ms` : (peaksLoading ? '…' : '—');
  const lossDisplay = peaks !== null ? `${peaks.packet_loss_pct.toFixed(1)}%` : (peaksLoading ? '…' : '—');
  const dnsDisplay = peaks !== null ? peaks.dns_failure_rate.toFixed(3) : (peaksLoading ? '…' : '—');
  const jitDisplay = peaks !== null ? `${Math.round(peaks.jitter_ms)}ms` : (peaksLoading ? '…' : '—');

  return (
    <div
      className={`
        p-3 rounded-lg border transition-all duration-200
        ${isNew ? 'animate-slide-in' : ''}
        ${alert.severity === 'critical' ? 'border-red-800/60 bg-red-900/10' :
          alert.severity === 'high' ? 'border-orange-800/60 bg-orange-900/10' :
          'border-gray-700 bg-gray-800/50'}
      `}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant={sev}>{alert.severity}</Badge>
          {alert.acknowledged && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-gray-500/20 text-gray-400 border border-gray-500/30 uppercase tracking-wide">
              ACK
            </span>
          )}
          <span className="text-sm font-medium text-gray-200">{alert.host_id}</span>
        </div>
        <span className="text-xs text-gray-500 shrink-0">
          {formatRelativeTime(alert.timestamp)}
        </span>
      </div>

      {/* Metric rows */}
      <div className="border border-gray-700/50 rounded-md px-3 py-1.5 bg-gray-900/40 mb-2">
        <MetricRow label="Latency" display={latDisplay} bad={latBad} />
        <MetricRow label="Packet Loss" display={lossDisplay} bad={lossBad} />
        <MetricRow label="DNS" display={dnsDisplay} bad={dnsBad} />
        <MetricRow label="Jitter" display={jitDisplay} bad={jitBad} />
      </div>

      {/* Anomaly type label */}
      {anomalyLabel && (
        <p className="text-xs font-medium text-gray-300 mb-2">{anomalyLabel}</p>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 mt-2">
        <button
          onClick={() => onAnalyze(alert)}
          className="flex-1 px-2 py-1.5 text-xs font-medium rounded-md bg-blue-600/20 text-blue-400 border border-blue-600/30 hover:bg-blue-600/30 transition-colors"
        >
          Analyze with AI
        </button>
        {!alert.acknowledged && (
          <button
            onClick={handleAcknowledge}
            disabled={acking}
            className="px-2 py-1.5 text-xs font-medium rounded-md bg-gray-700 text-gray-300 border border-gray-600 hover:bg-gray-600 transition-colors disabled:opacity-50"
          >
            {acking ? '…' : 'Acknowledge'}
          </button>
        )}
      </div>
    </div>
  );
}

// ─── AlertFeed ─────────────────────────────────────────────────────────────────

interface AlertFeedProps {
  /** Called when the user clicks "Analyze with AI" on an alert card. */
  onViewAnomalies: (alert: AlertRecord) => void;
}

/**
 * Live-scrolling feed of fired alerts with per-metric health indicators.
 *
 * - Auto-refreshes every 5 s via /alerts/recent.
 * - Fetches 5-minute peak telemetry per unique host to populate metric rows.
 * - "Analyze with AI" opens AlertDetailModal for drill-down.
 * - "Acknowledge" calls PUT /alerts/{alert_id}/acknowledge.
 */
export function AlertFeed({ onViewAnomalies }: AlertFeedProps) {
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const [peaksByHost, setPeaksByHost] = useState<Record<string, HostPeaks>>({});
  const [loadingHosts, setLoadingHosts] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const prevIdsRef = useRef<Set<string>>(new Set());
  const fetchedHostsRef = useRef<Set<string>>(new Set());
  const mountedRef = useRef(true);

  async function fetchAlerts() {
    try {
      const data = await getRecentAlerts(undefined, undefined, 30);
      if (!mountedRef.current) return;

      const incoming = data.alerts;
      const incomingIds = new Set(incoming.map((a) => a.alert_id));
      const fresh = new Set<string>();
      incomingIds.forEach((id) => {
        if (!prevIdsRef.current.has(id)) fresh.add(id);
      });

      prevIdsRef.current = incomingIds;
      setAlerts(incoming);
      setNewIds(fresh);
      setError(null);

      if (fresh.size > 0) {
        setTimeout(() => { if (mountedRef.current) setNewIds(new Set()); }, 800);
      }

      // Fetch telemetry peaks for any new hosts
      const uniqueHosts = [...new Set(incoming.map((a) => a.host_id))];
      const newHosts = uniqueHosts.filter((h) => !fetchedHostsRef.current.has(h));
      if (newHosts.length > 0) {
        setLoadingHosts((prev) => new Set([...prev, ...newHosts]));
        newHosts.forEach((hostId) => {
          fetchedHostsRef.current.add(hostId);
          getRecentTelemetry(hostId, 5, 100)
            .then((telData) => {
              if (!mountedRef.current) return;
              const peaks = computePeaks(telData.records);
              setPeaksByHost((prev) => ({ ...prev, [hostId]: peaks }));
            })
            .catch(() => {/* silently ignore — metric rows will show "—" */})
            .finally(() => {
              if (mountedRef.current) {
                setLoadingHosts((prev) => {
                  const next = new Set(prev);
                  next.delete(hostId);
                  return next;
                });
              }
            });
        });
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load alerts');
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }

  function handleAcknowledge(alertId: string) {
    setAlerts((prev) =>
      prev.map((a) => a.alert_id === alertId ? { ...a, acknowledged: true } : a)
    );
  }

  useEffect(() => {
    mountedRef.current = true;
    fetchAlerts();
    const id = setInterval(fetchAlerts, POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const headerAction = alerts.length > 0 && (
    <span className="text-xs text-gray-500">{alerts.length} alerts</span>
  );

  if (loading) return <CardSkeleton className="h-72" />;

  return (
    <Card
      title="Alert Feed"
      subtitle="Peak metrics over the last 5 minutes"
      headerAction={headerAction}
      className="h-full flex flex-col"
      bodyClassName="flex-1 flex flex-col min-h-0 overflow-hidden"
    >
      {error && <AlertBanner level="error" message={error} className="shrink-0 mb-3" />}

      <div className="flex-1 min-h-0 overflow-y-auto space-y-2 pr-1">
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-44 text-center">
            <span className="text-2xl mb-2">✓</span>
            <p className="text-gray-400 text-sm font-medium">No active alerts</p>
            <p className="text-gray-600 text-xs mt-1">System healthy</p>
          </div>
        ) : (
          alerts.map((a) => (
            <AlertItem
              key={a.alert_id}
              alert={a}
              isNew={newIds.has(a.alert_id)}
              peaks={peaksByHost[a.host_id] ?? null}
              peaksLoading={loadingHosts.has(a.host_id)}
              onAnalyze={onViewAnomalies}
              onAcknowledge={handleAcknowledge}
            />
          ))
        )}
      </div>
    </Card>
  );
}
