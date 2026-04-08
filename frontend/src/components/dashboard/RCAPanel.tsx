'use client';

import { useEffect, useRef, useState } from 'react';
import { analyzeAnomaly, chatWithAssistant, type AnomalyRecord, type ChatMessage } from '@/lib/api';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { AlertBanner } from '@/components/ui/Alert';

const MAX_HISTORY = 10;

// ─── Markdown stripper ─────────────────────────────────────────────────────────

/**
 * Remove common markdown syntax so raw LLM output renders as clean text.
 * Handles: **bold**, *italic*, ## headings, __underline__.
 * Bullet markers (- / • / *) inside action lists are preserved by the
 * caller — this function only strips inline and block-level decoration.
 */
function cleanMarkdown(text: string): string {
  return text
    // Strip ## / ### headings — keep the heading text, drop the # prefix
    .replace(/^#{1,6}\s+/gm, '')
    // Strip **bold** markers
    .replace(/\*\*[\s\S]+?\*\*/g, (m) => m.slice(2, -2))
    // Strip *italic* markers (single asterisk, not bullet at line start)
    .replace(/(?<!\n)\*[\s\S]+?\*/g, (m) => m.slice(1, -1))
    // Strip __underline__ markers
    .replace(/__[\s\S]+?__/g, (m) => m.slice(2, -2))
    // Collapse 3+ consecutive blank lines down to one
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ─── Section parser ────────────────────────────────────────────────────────────

interface ParsedRCA {
  whatIsHappening: string;
  rootCause: string;
  immediateActions: string[];
  severityJustification: string;
  parsed: boolean;
}

function parseAnalysis(text: string, apiActions: string[]): ParsedRCA {
  const matches = [...text.matchAll(/^\s*(\d+)\.\s+/gm)];

  if (matches.length < 2) {
    return {
      whatIsHappening: cleanMarkdown(text),
      rootCause: '',
      immediateActions: apiActions.map(cleanMarkdown),
      severityJustification: '',
      parsed: false,
    };
  }

  const sections: Record<number, string> = {};
  matches.forEach((m, idx) => {
    const num = parseInt(m[1], 10);
    const contentStart = m.index! + m[0].length;
    const contentEnd = idx + 1 < matches.length ? matches[idx + 1].index! : text.length;
    // Strip the section heading line (e.g. "What is happening:") if present
    const raw = text.slice(contentStart, contentEnd).trim();
    // Remove a leading "heading:" line (first line ending with colon) if any
    const lines = raw.split('\n');
    const body = lines[0].trim().endsWith(':') ? lines.slice(1).join('\n').trim() : raw;
    sections[num] = body;
  });

  // Parse bullet actions from section 3 if API didn't return them
  let actions = apiActions.map(cleanMarkdown);
  if (actions.length === 0 && sections[3]) {
    const bullets = [...sections[3].matchAll(/^\s*[-•*]\s+(.+)$/gm)].map((m) => cleanMarkdown(m[1].trim()));
    actions = bullets.length > 0 ? bullets : sections[3].split('\n').map((l) => cleanMarkdown(l.trim())).filter(Boolean);
  }

  return {
    whatIsHappening: cleanMarkdown(sections[1] ?? text.slice(0, 300)),
    rootCause: cleanMarkdown(sections[2] ?? ''),
    immediateActions: actions,
    severityJustification: cleanMarkdown(sections[4] ?? ''),
    parsed: !!(sections[1] && sections[2]),
  };
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-3 py-2 bg-gray-700 rounded-xl rounded-tl-sm w-fit">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div
        className={`
          max-w-[85%] px-3 py-2 rounded-xl text-sm leading-relaxed whitespace-pre-wrap
          ${isUser
            ? 'bg-blue-600 text-white rounded-tr-sm'
            : 'bg-gray-700 text-gray-100 rounded-tl-sm'}
        `}
      >
        {msg.content}
      </div>
    </div>
  );
}

function SectionBox({
  icon,
  title,
  borderColor,
  children,
}: {
  icon: string;
  title: string;
  borderColor: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`rounded-lg bg-gray-800/80 border border-gray-700 border-l-2 ${borderColor} p-3 mb-2`}>
      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1.5">
        {icon} {title}
      </p>
      {children}
    </div>
  );
}

function ConfidenceBar({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500';
  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="text-xs text-gray-500 shrink-0">AI Confidence:</span>
      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-semibold text-gray-300 w-8 text-right tabular-nums">{pct}%</span>
    </div>
  );
}

function ActionChecklist({ actions }: { actions: string[] }) {
  const [checked, setChecked] = useState<Set<number>>(new Set());

  function toggle(i: number) {
    setChecked((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  }

  return (
    <ul className="space-y-1.5">
      {actions.map((action, i) => (
        <li key={i} className="flex items-start gap-2">
          <button
            onClick={() => toggle(i)}
            className={`
              mt-0.5 w-4 h-4 flex-shrink-0 rounded border transition-colors
              ${checked.has(i) ? 'bg-green-500 border-green-500 text-white' : 'border-gray-500 hover:border-gray-400'}
            `}
            aria-label={`Mark action ${i + 1} done`}
          >
            {checked.has(i) && <span className="block text-center text-xs leading-4">✓</span>}
          </button>
          <span className={`text-xs leading-relaxed ${checked.has(i) ? 'line-through text-gray-500' : 'text-gray-300'}`}>
            {action}
          </span>
        </li>
      ))}
    </ul>
  );
}

function RCAResult({
  analysis,
  actions,
  confidence,
  severity,
}: {
  analysis: string;
  actions: string[];
  confidence: number;
  severity: string;
}) {
  const rca = parseAnalysis(analysis, actions);

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 mb-2">
        <Badge variant={severity as 'critical' | 'high' | 'medium' | 'low'}>{severity}</Badge>
      </div>

      <ConfidenceBar confidence={confidence} />

      {rca.parsed ? (
        <>
          {rca.whatIsHappening && (
            <SectionBox icon="🔍" title="What is Happening" borderColor="border-l-blue-500">
              <p className="text-xs text-gray-200 leading-relaxed">{rca.whatIsHappening}</p>
            </SectionBox>
          )}

          {rca.rootCause && (
            <SectionBox icon="🧠" title="Root Cause" borderColor="border-l-purple-500">
              <p className="text-xs text-gray-200 leading-relaxed">{rca.rootCause}</p>
            </SectionBox>
          )}

          {rca.immediateActions.length > 0 && (
            <SectionBox icon="⚡" title="Immediate Actions" borderColor="border-l-red-500">
              <ActionChecklist actions={rca.immediateActions} />
            </SectionBox>
          )}

          {rca.severityJustification && (
            <SectionBox icon="⚠️" title="Severity" borderColor="border-l-orange-500">
              <p className="text-xs text-gray-200 leading-relaxed">{rca.severityJustification}</p>
            </SectionBox>
          )}
        </>
      ) : (
        <>
          <SectionBox icon="🔍" title="Analysis" borderColor="border-l-blue-500">
            <p className="text-xs text-gray-200 leading-relaxed whitespace-pre-wrap">{analysis}</p>
          </SectionBox>
          {actions.length > 0 && (
            <SectionBox icon="⚡" title="Immediate Actions" borderColor="border-l-red-500">
              <ActionChecklist actions={actions} />
            </SectionBox>
          )}
        </>
      )}
    </div>
  );
}

// ─── RCAPanel ──────────────────────────────────────────────────────────────────

interface RCAPanelProps {
  /** Anomaly selected in the AlertDetailModal to pre-populate the panel. */
  selectedAnomaly: AnomalyRecord | null;
  /** Latest anomaly in the system for the quick-action button. */
  latestAnomaly: AnomalyRecord | null;
}

/**
 * AI assistant chat panel for root cause analysis.
 *
 * - Maintains up to 10 messages of conversation history.
 * - "Analyze latest anomaly" quick action button calls /assistant/analyze.
 * - Free-text chat calls /assistant/chat with history context.
 * - Shows typing indicator while waiting for API response.
 * - Renders structured RCA results in color-coded section boxes.
 */
export function RCAPanel({ selectedAnomaly, latestAnomaly }: RCAPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        'Hello! I\'m FlowWatch AI. Click "Analyze with AI" on an alert to start investigating network issues.',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [rcaResult, setRcaResult] = useState<{
    analysis: string;
    actions: string[];
    confidence: number;
    severity: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const analysedIdRef = useRef<string | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    if (selectedAnomaly && selectedAnomaly.record_id !== analysedIdRef.current) {
      analysedIdRef.current = selectedAnomaly.record_id;
      runAnalysis(selectedAnomaly);
    }
  }, [selectedAnomaly]); // eslint-disable-line react-hooks/exhaustive-deps

  async function runAnalysis(anomaly: AnomalyRecord) {
    setIsLoading(true);
    setError(null);
    setRcaResult(null);

    const userMsg: ChatMessage = {
      role: 'user',
      content: `Analyze anomaly on ${anomaly.host_id} (${anomaly.severity}, score: ${anomaly.combined_score.toFixed(3)})`,
    };
    setMessages((prev) => [...prev.slice(-(MAX_HISTORY - 1)), userMsg]);

    try {
      const result = await analyzeAnomaly({
        host_id: anomaly.host_id,
        anomaly_result: anomaly as unknown as Record<string, unknown>,
        recent_telemetry: [],
        question: 'What is causing this anomaly and what should I do?',
      });

      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: `Analysis complete for ${anomaly.host_id}.`,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setRcaResult({
        analysis: result.analysis,
        actions: result.recommended_actions,
        confidence: result.confidence,
        severity: result.anomaly_severity,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsLoading(false);
    }
  }

  async function sendChat() {
    const text = input.trim();
    if (!text || isLoading) return;

    setInput('');
    setIsLoading(true);
    setError(null);

    const userMsg: ChatMessage = { role: 'user', content: text };
    const updatedHistory = [...messages.slice(-(MAX_HISTORY - 1)), userMsg];
    setMessages(updatedHistory);

    try {
      const result = await chatWithAssistant(text, updatedHistory);
      setMessages(result.conversation_history.slice(-MAX_HISTORY));
      setRcaResult(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chat failed');
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  }

  return (
    <Card
      title="AI Assistant"
      subtitle="Claude-powered root cause analysis"
      className="h-full flex flex-col"
      bodyClassName="flex-1 flex flex-col min-h-0 overflow-hidden"
    >
      {error && <AlertBanner level="error" message={error} className="shrink-0 mb-3" />}

      {/* Message history — grows to fill all available card space */}
      <div className="flex-1 min-h-0 overflow-y-auto mb-3 pr-1">
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
        {isLoading && (
          <div className="flex justify-start mb-3">
            <TypingIndicator />
          </div>
        )}
        {rcaResult && !isLoading && (
          <div className="mt-2">
            <RCAResult {...rcaResult} />
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick action — shrink-0 so it never scrolls out of view */}
      {latestAnomaly && (
        <button
          onClick={() => runAnalysis(latestAnomaly)}
          disabled={isLoading}
          className="
            shrink-0 w-full mb-2 px-3 py-2 text-xs font-medium rounded-lg
            bg-blue-600/20 text-blue-400 border border-blue-600/30
            hover:bg-blue-600/30 transition-colors
            disabled:opacity-50 disabled:cursor-not-allowed
          "
        >
          ⚡ Analyze Latest Anomaly ({latestAnomaly.host_id} — {latestAnomaly.severity})
        </button>
      )}

      {/* Chat input — always pinned to bottom */}
      <div className="shrink-0 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          placeholder="Ask about network health…"
          className="
            flex-1 px-3 py-2 text-sm rounded-lg
            bg-gray-700 border border-gray-600 text-gray-100
            placeholder-gray-500 outline-none
            focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30
            disabled:opacity-50 transition-colors
          "
        />
        <button
          onClick={sendChat}
          disabled={isLoading || !input.trim()}
          className="
            px-3 py-2 text-sm font-medium rounded-lg
            bg-blue-600 text-white
            hover:bg-blue-700 transition-colors
            disabled:opacity-50 disabled:cursor-not-allowed
          "
          aria-label="Send message"
        >
          ➤
        </button>
      </div>
    </Card>
  );
}
