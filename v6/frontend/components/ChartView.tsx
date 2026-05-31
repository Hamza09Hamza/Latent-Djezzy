"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { ChartSpec } from "@/lib/api";

// Theme — teal primary, amber accent, then a spread for multi-series.
const COLORS = [
  "hsl(195 70% 55%)",
  "hsl(45 85% 65%)",
  "hsl(265 60% 68%)",
  "hsl(150 55% 55%)",
  "hsl(330 65% 68%)",
  "hsl(20 80% 62%)",
];
const AXIS = "hsl(0 0% 55%)";
const GRID = "hsl(0 0% 22%)";

function strip(x: number): string {
  return (Math.round(x * 100) / 100).toString();
}

/** Compact axis ticks (1.2M, 253K) — tooltips use the precise frozen string. */
function compact(n: unknown): string {
  const v = typeof n === "number" ? n : Number(n);
  if (!isFinite(v)) return String(n ?? "");
  const a = Math.abs(v);
  if (a >= 1e9) return strip(v / 1e9) + "B";
  if (a >= 1e6) return strip(v / 1e6) + "M";
  if (a >= 1e3) return strip(v / 1e3) + "K";
  return strip(v);
}

function firstFmt(d: Record<string, unknown>): string | undefined {
  const k = Object.keys(d).find((key) => key.endsWith("_fmt"));
  return k ? (d[k] as string) : undefined;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-popover/95 px-3 py-2 shadow-lg backdrop-blur-sm">
      {label != null && label !== "" && (
        <div className="text-xs text-muted-foreground mb-1">{String(label)}</div>
      )}
      <div className="space-y-0.5">
        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
        {payload.map((it: any, i: number) => {
          const datum = (it.payload || {}) as Record<string, unknown>;
          const fmt =
            (datum[`${it.dataKey}_fmt`] as string) ??
            firstFmt(datum) ??
            compact(it.value);
          return (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span
                className="inline-block w-2 h-2 rounded-sm flex-shrink-0"
                style={{ background: it.color || it.fill || it.stroke }}
              />
              <span className="text-muted-foreground">{it.name}</span>
              <span className="ml-auto font-medium text-foreground">{fmt}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const AXIS_TICK = { fontSize: 11, fill: AXIS } as const;

export function ChartView({ spec }: { spec: ChartSpec }) {
  if (!spec?.data?.length) return null;
  const { type, series, data, x } = spec;
  const multi = series.length > 1;

  const header = (
    <div className="mb-2">
      {spec.title && (
        <div className="text-sm font-medium text-foreground leading-snug">
          {spec.title}
        </div>
      )}
      {series[0]?.unit && type !== "stat" && (
        <div className="text-[11px] text-muted-foreground">
          in {series[0].unit}
        </div>
      )}
    </div>
  );

  // ── single KPI → a big stat card ──────────────────────────────────────
  if (type === "stat") {
    const d = data[0] || {};
    const s = series[0];
    const value = (d[`${s.key}_fmt`] as string) ?? String(d[s.key] ?? "—");
    return (
      <div className="rounded-xl border border-border/60 bg-background/40 p-5">
        {header}
        <div className="py-3 text-center">
          <div className="text-3xl font-semibold text-primary">{value}</div>
          <div className="text-xs text-muted-foreground mt-1">{s.label}</div>
        </div>
      </div>
    );
  }

  let chart: React.ReactNode = null;

  if (type === "line" || type === "area") {
    const Comp = type === "area" ? AreaChart : LineChart;
    chart = (
      <Comp data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
        <defs>
          {series.map((s, i) => (
            <linearGradient
              key={s.key}
              id={`grad-${i}`}
              x1="0"
              y1="0"
              x2="0"
              y2="1"
            >
              <stop offset="0%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0.35} />
              <stop offset="100%" stopColor={COLORS[i % COLORS.length]} stopOpacity={0.02} />
            </linearGradient>
          ))}
        </defs>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey={x?.key} stroke={GRID} tick={AXIS_TICK} tickMargin={8} />
        <YAxis
          stroke={GRID}
          tick={AXIS_TICK}
          tickFormatter={compact}
          width={52}
        />
        <Tooltip content={<ChartTooltip />} cursor={{ stroke: GRID }} />
        {multi && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s, i) =>
          type === "area" ? (
            <Area
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              fill={`url(#grad-${i})`}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ) : (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          )
        )}
      </Comp>
    );
  } else if (type === "bar") {
    chart = (
      <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey={x?.key} stroke={GRID} tick={AXIS_TICK} tickMargin={8} />
        <YAxis stroke={GRID} tick={AXIS_TICK} tickFormatter={compact} width={52} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "hsl(0 0% 100% / 0.04)" }} />
        {multi && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {series.map((s, i) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            name={s.label}
            radius={[4, 4, 0, 0]}
            fill={COLORS[i % COLORS.length]}
          />
        ))}
      </BarChart>
    );
  } else if (type === "hbar") {
    const s = series[0];
    chart = (
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 8, right: 16, left: 8, bottom: 4 }}
      >
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" stroke={GRID} tick={AXIS_TICK} tickFormatter={compact} />
        <YAxis
          type="category"
          dataKey={x?.key}
          stroke={GRID}
          tick={AXIS_TICK}
          width={120}
        />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "hsl(0 0% 100% / 0.04)" }} />
        <Bar dataKey={s.key} name={s.label} radius={[0, 4, 4, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    );
  } else if (type === "pie") {
    const s = series[0];
    chart = (
      <PieChart>
        <Pie
          data={data}
          dataKey={s.key}
          nameKey={x?.key}
          innerRadius={62}
          outerRadius={100}
          paddingAngle={2}
          stroke="hsl(0 0% 12%)"
        >
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
      </PieChart>
    );
  }

  const height =
    type === "hbar" ? Math.max(200, data.length * 38 + 40) : 300;

  return (
    <div className="rounded-xl border border-border/60 bg-background/40 p-4">
      {header}
      <ResponsiveContainer width="100%" height={height}>
        {chart as React.ReactElement}
      </ResponsiveContainer>
    </div>
  );
}
