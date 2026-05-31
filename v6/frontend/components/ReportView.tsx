"use client";

import { useState } from "react";
import {
  Printer,
  Mail,
  TrendingUp,
  TrendingDown,
  Minus,
  Send,
  CheckCircle2,
} from "lucide-react";
import { ChartView } from "@/components/ChartView";
import { ReportSpec, ReportSection, ReportMetricItem } from "@/lib/api";

// ── Metric chip ──────────────────────────────────────────────────────────────

function MetricCard({ item }: { item: ReportMetricItem }) {
  const Icon =
    item.trend === "up"
      ? TrendingUp
      : item.trend === "down"
      ? TrendingDown
      : Minus;
  const trendColor =
    item.trend === "up"
      ? "text-primary"
      : item.trend === "down"
      ? "text-destructive"
      : "text-muted-foreground";
  return (
    <div className="flex-1 min-w-[130px] rounded-lg border border-border/50 bg-card/60 px-3 py-2.5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">
        {item.label}
      </p>
      <p className="text-sm font-bold text-foreground leading-tight">{item.value}</p>
      {item.change && (
        <p className={`text-[10px] flex items-center gap-0.5 mt-1 ${trendColor}`}>
          <Icon className="w-3 h-3" />
          {item.change}
        </p>
      )}
    </div>
  );
}

// ── Data table ───────────────────────────────────────────────────────────────

function TableSection({
  columns,
  rows,
}: {
  columns: string[];
  rows: (string | number)[][];
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border/40">
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-muted/30 border-b border-border/40">
            {columns.map((col, i) => (
              <th
                key={i}
                className="px-3 py-2 text-left font-semibold text-foreground/70"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr
              key={ri}
              className={
                ri % 2 === 0 ? "bg-background/10" : "bg-muted/10"
              }
            >
              {row.map((cell, ci) => (
                <td key={ci} className="px-3 py-2 text-foreground/80 font-mono">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Section dispatcher ───────────────────────────────────────────────────────

function Section({ s }: { s: ReportSection }) {
  if (s.type === "heading") {
    return (
      <h3 className="text-xs font-semibold text-foreground/90 uppercase tracking-wider border-b border-border/30 pb-1 pt-2 first:pt-0">
        {s.text}
      </h3>
    );
  }
  if (s.type === "text") {
    return (
      <p className="text-xs text-foreground/80 leading-relaxed">{s.text}</p>
    );
  }
  if (s.type === "chart" && s.spec) {
    return <ChartView spec={s.spec} />;
  }
  if (s.type === "metrics" && s.items) {
    return (
      <div className="flex flex-wrap gap-2">
        {s.items.map((item, i) => (
          <MetricCard key={i} item={item} />
        ))}
      </div>
    );
  }
  if (s.type === "table" && s.columns && s.rows) {
    return <TableSection columns={s.columns} rows={s.rows} />;
  }
  return null;
}

// ── Print-to-PDF helper (opens clean popup, auto-prints) ─────────────────────

function buildPrintHtml(spec: ReportSpec): string {
  const dateStr = spec.generated_at
    ? new Date(spec.generated_at).toLocaleString()
    : new Date().toLocaleString();

  const sectionsHtml = spec.sections
    .map((s) => {
      if (s.type === "heading")
        return `<h3>${s.text}</h3>`;
      if (s.type === "text")
        return `<p>${s.text}</p>`;
      if (s.type === "metrics" && s.items)
        return `<div class="metrics">${s.items
          .map(
            (it) =>
              `<div class="metric"><div class="ml">${it.label}</div><div class="mv">${it.value}</div>${it.change ? `<div class="mc">${it.change}</div>` : ""}</div>`
          )
          .join("")}</div>`;
      if (s.type === "table" && s.columns && s.rows)
        return `<table><thead><tr>${s.columns
          .map((c) => `<th>${c}</th>`)
          .join("")}</tr></thead><tbody>${s.rows
          .map(
            (r, ri) =>
              `<tr class="${ri % 2 === 0 ? "even" : ""}">${r
                .map((cell) => `<td>${cell}</td>`)
                .join("")}</tr>`
          )
          .join("")}</tbody></table>`;
      if (s.type === "chart")
        return `<p class="chart-placeholder"><em>[Chart: ${s.spec?.title ?? ""}]</em></p>`;
      return "";
    })
    .join("\n");

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>${spec.title}</title>
<style>
  body{font-family:system-ui,sans-serif;padding:40px 48px;color:#111;background:#fff;max-width:860px;margin:0 auto}
  h1{font-size:20px;font-weight:700;margin:0 0 4px}
  .sub{font-size:11px;color:#888;margin-bottom:28px}
  h3{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #e0e0e0;padding-bottom:4px;margin:24px 0 10px;color:#555}
  p{font-size:12px;line-height:1.65;margin:8px 0;color:#333}
  .metrics{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}
  .metric{border:1px solid #ddd;border-radius:8px;padding:10px 14px;min-width:130px}
  .ml{font-size:9px;text-transform:uppercase;letter-spacing:.05em;color:#888;margin-bottom:3px}
  .mv{font-size:15px;font-weight:700;color:#111}
  .mc{font-size:10px;color:#2a9d8f;margin-top:2px}
  table{width:100%;border-collapse:collapse;margin:8px 0;font-size:11px}
  th{background:#f5f5f5;text-align:left;padding:7px 12px;font-weight:600;border-bottom:2px solid #ddd;font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:#555}
  td{padding:6px 12px;border-bottom:1px solid #f0f0f0;font-family:monospace;color:#333}
  tr.even td{background:#fafafa}
  .chart-placeholder{color:#aaa;font-style:italic}
  @media print{body{padding:0}button{display:none}}
</style>
</head>
<body>
<h1>${spec.title}</h1>
<p class="sub">${spec.subtitle ? spec.subtitle + " &middot; " : ""}${dateStr}</p>
${sectionsHtml}
<script>window.onload=()=>{window.print();window.onafterprint=()=>window.close()}<\/script>
</body>
</html>`;
}

// ── ReportView ───────────────────────────────────────────────────────────────

interface ReportViewProps {
  spec: ReportSpec;
  onEmail?: (to: string) => void;
}

export function ReportView({ spec, onEmail }: ReportViewProps) {
  const [emailOpen, setEmailOpen] = useState(false);
  const [emailTo, setEmailTo] = useState("");
  const [emailSent, setEmailSent] = useState(false);

  const dateStr = spec.generated_at
    ? new Date(spec.generated_at).toLocaleString()
    : "";

  function handlePrint() {
    const html = buildPrintHtml(spec);
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank", "width=860,height=960,resizable=yes");
    setTimeout(() => URL.revokeObjectURL(url), 90_000);
  }

  function handleSendEmail() {
    if (!emailTo.trim()) return;
    onEmail?.(emailTo.trim());
    setEmailSent(true);
    setTimeout(() => {
      setEmailSent(false);
      setEmailOpen(false);
      setEmailTo("");
    }, 2500);
  }

  return (
    <div className="rounded-lg border border-border/40 bg-card/40 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-start justify-between gap-3 px-4 py-3 border-b border-border/30 bg-muted/10">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground leading-tight">
            {spec.title}
          </h2>
          {(spec.subtitle || dateStr) && (
            <p className="text-[10px] text-muted-foreground mt-0.5">
              {spec.subtitle}
              {spec.subtitle && dateStr ? " · " : ""}
              {dateStr}
            </p>
          )}
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button
            onClick={() => { setEmailOpen((o) => !o); setEmailSent(false); }}
            className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-primary border border-border/50 rounded px-2 py-1 transition-colors"
          >
            <Mail className="w-3 h-3" /> Email
          </button>
          <button
            onClick={handlePrint}
            className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground border border-border/50 rounded px-2 py-1 transition-colors"
          >
            <Printer className="w-3 h-3" /> PDF
          </button>
        </div>
      </div>

      {/* Inline email form */}
      {emailOpen && (
        <div className="px-4 py-2.5 border-b border-border/20 bg-primary/5 flex items-center gap-2">
          {emailSent ? (
            <p className="text-xs text-primary flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" /> Report queued for delivery to {emailTo}
            </p>
          ) : (
            <>
              <Mail className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
              <input
                type="email"
                value={emailTo}
                onChange={(e) => setEmailTo(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSendEmail()}
                placeholder="recipient@example.com"
                className="flex-1 bg-transparent text-xs text-foreground placeholder:text-muted-foreground border-none outline-none"
              />
              <button
                onClick={handleSendEmail}
                disabled={!emailTo.trim()}
                className="flex items-center gap-1 text-xs text-primary hover:text-primary/80 disabled:opacity-40"
              >
                <Send className="w-3 h-3" /> Send
              </button>
            </>
          )}
        </div>
      )}

      {/* Report body */}
      <div className="px-4 py-3 space-y-2.5">
        {spec.sections.map((s, i) => (
          <Section key={i} s={s} />
        ))}
      </div>
    </div>
  );
}
