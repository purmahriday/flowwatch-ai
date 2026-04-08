/**
 * FlowWatch AI — Centralised API Client
 * ======================================
 * All fetch calls to the FastAPI backend live here. Components import
 * typed async functions; no ad-hoc fetch calls appear in component files.
 *
 * Authentication: every request carries the X-API-Key header.
 * Base URL and API key come from NEXT_PUBLIC_* environment variables.
 */

// ─── Configuration ────────────────────────────────────────────────────────────

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? '';

// ─── TypeScript interfaces ─────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  uptime_seconds: number;
  models_loaded: boolean;
  lstm_version: string;
  if_version: string;
  total_records_processed: number;
}

export interface LSTMResultSchema {
  is_anomaly: boolean;
  anomaly_score: number;
  reconstruction_error: number;
  threshold_used: number;
  per_feature_errors: Record<string, number>;
  worst_feature: string;
  inference_time_ms: number;
  model_version: string;
}

export interface IFResultSchema {
  is_anomaly: boolean;
  anomaly_score: number;
  raw_score: number;
  confidence: number;
  top_contributing_features: string[];
  host_id: string;
  timestamp: string;
  model_version: string;
  inference_time_ms: number;
}

export interface CombinedAnomalyResultSchema {
  is_anomaly: boolean;
  combined_score: number;
  severity: string;
  lstm_result: LSTMResultSchema;
  if_result: IFResultSchema;
  detection_method: string;
  worst_feature: string;
  top_contributing_features: string[];
  timestamp: string;
}

export interface TelemetryIngestRequest {
  host_id: string;
  latency_ms: number;
  packet_loss_pct: number;
  dns_failure_rate: number;
  jitter_ms: number;
  timestamp?: string;
}

export interface TelemetryIngestResponse {
  record_id: string;
  host_id: string;
  processed: boolean;
  anomaly_detected: boolean;
  anomaly_result: CombinedAnomalyResultSchema | null;
  window_ready: boolean;
  message: string;
  alert_id: string | null;
}

export interface TelemetryRecord {
  host_id: string;
  timestamp: string;
  latency_ms: number;
  packet_loss_pct: number;
  dns_failure_rate: number;
  jitter_ms: number;
  composite_health_score: number;
  latency_normalized: number;
  loss_normalized: number;
  is_business_hours: boolean;
}

export interface TelemetryRecentResponse {
  records: TelemetryRecord[];
  total_count: number;
  hosts_included: string[];
  time_range_minutes: number;
}

export interface HostStatusResponse {
  host_id: string;
  record_count: number;
  latest_health_score: number;
  latest_latency_ms: number;
  window_ready: boolean;
  last_seen: string;
}

export interface AnomalyRecord {
  record_id: string;
  host_id: string;
  detected_at: string;
  severity: string;
  combined_score: number;
  is_anomaly: boolean;
  detection_method: string;
  worst_feature: string;
  top_contributing_features: string[];
  lstm_result: LSTMResultSchema;
  if_result: IFResultSchema;
  anomaly_timestamp: string;
  recommendation: string;
}

export interface AnomalyListResponse {
  anomalies: AnomalyRecord[];
  total_count: number;
  critical_count: number;
  high_count: number;
  summary_by_host: Record<string, number>;
}

export interface AnomalyStatsResponse {
  total_anomalies_detected: number;
  anomaly_rate_last_hour: number;
  most_affected_host: string;
  most_common_severity: string;
  worst_feature_overall: string;
  detection_breakdown: Record<string, number>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AssistantAnalyzeRequest {
  host_id: string;
  anomaly_result: Record<string, unknown>;
  recent_telemetry: Record<string, unknown>[];
  question?: string;
}

export interface AssistantAnalyzeResponse {
  host_id: string;
  analysis: string;
  anomaly_severity: string;
  recommended_actions: string[];
  confidence: number;
  analysis_timestamp: string;
  model_used: string;
}

export interface AssistantChatRequest {
  host_id: string;
  message: string;
  conversation_history: ChatMessage[];
}

export interface AssistantChatResponse {
  host_id: string;
  response: string;
  conversation_history: ChatMessage[];
}

export interface AlertRecord {
  alert_id: string;
  host_id: string;
  severity: string;
  combined_score: number;
  worst_feature: string;
  top_contributing_features: string[];
  message: string;
  timestamp: string;
  acknowledged: boolean;
  resolved: boolean;
  resolution_timestamp: string | null;
}

export interface AlertRecentResponse {
  alerts: AlertRecord[];
  total_count: number;
  filters_applied: Record<string, string>;
}

export interface AlertStatsResponse {
  total_alerts_fired: number;
  alerts_suppressed: number;
  alerts_by_severity: Record<string, number>;
  alerts_by_host: Record<string, number>;
  most_affected_host: string;
  last_alert_timestamp: string | null;
}

// ─── HTTP helper ──────────────────────────────────────────────────────────────

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
      ...options.headers,
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }

  return res.json() as Promise<T>;
}

// ─── API functions ─────────────────────────────────────────────────────────────

/** GET /health — liveness + readiness probe. No API key required. */
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json() as Promise<HealthResponse>;
}

/** POST /telemetry/ingest — submit a raw telemetry record. */
export async function ingestTelemetry(
  record: TelemetryIngestRequest,
): Promise<TelemetryIngestResponse> {
  return request<TelemetryIngestResponse>('/telemetry/ingest', {
    method: 'POST',
    body: JSON.stringify(record),
  });
}

/** GET /telemetry/recent — fetch recent ProcessedRecords. */
export async function getRecentTelemetry(
  hostId?: string,
  minutes = 5,
  limit = 300,
): Promise<TelemetryRecentResponse> {
  const params = new URLSearchParams({ minutes: String(minutes), limit: String(limit) });
  if (hostId) params.set('host_id', hostId);
  return request<TelemetryRecentResponse>(`/telemetry/recent?${params}`);
}

/** GET /telemetry/hosts — health snapshot per host. */
export async function getHosts(): Promise<HostStatusResponse[]> {
  return request<HostStatusResponse[]>('/telemetry/hosts');
}

/** GET /anomalies/latest — recent anomaly records. */
export async function getLatestAnomalies(
  hostId?: string,
  severity?: string,
  limit = 50,
): Promise<AnomalyListResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (hostId) params.set('host_id', hostId);
  if (severity) params.set('severity', severity);
  return request<AnomalyListResponse>(`/anomalies/latest?${params}`);
}

/** GET /anomalies/stats — aggregate anomaly statistics. */
export async function getAnomalyStats(): Promise<AnomalyStatsResponse> {
  return request<AnomalyStatsResponse>('/anomalies/stats');
}

/** POST /assistant/analyze — LLM root cause analysis. */
export async function analyzeAnomaly(
  req: AssistantAnalyzeRequest,
): Promise<AssistantAnalyzeResponse> {
  return request<AssistantAnalyzeResponse>('/assistant/analyze', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/** GET /alerts/recent — fetch recent fired alerts. */
export async function getRecentAlerts(
  hostId?: string,
  severity?: string,
  limit = 50,
): Promise<AlertRecentResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (hostId) params.set('host_id', hostId);
  if (severity) params.set('severity', severity);
  return request<AlertRecentResponse>(`/alerts/recent?${params}`);
}

/** GET /alerts/stats — aggregate alert statistics. */
export async function getAlertStats(): Promise<AlertStatsResponse> {
  return request<AlertStatsResponse>('/alerts/stats');
}

/** PUT /alerts/{alert_id}/acknowledge — mark an alert acknowledged. */
export async function acknowledgeAlert(alertId: string): Promise<void> {
  await request<unknown>(`/alerts/${alertId}/acknowledge`, { method: 'PUT' });
}

/** POST /assistant/chat — multi-turn conversation. */
export async function chatWithAssistant(
  message: string,
  history: ChatMessage[],
  hostId = 'dashboard',
): Promise<AssistantChatResponse> {
  const body: AssistantChatRequest = {
    host_id: hostId,
    message,
    conversation_history: history,
  };
  return request<AssistantChatResponse>('/assistant/chat', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
