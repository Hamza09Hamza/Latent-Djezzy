// ============================================================================
//  Demo data — a fully faked conversation for the /demo page.
//
//  No backend, no models: this is canned content so you can see how the UI
//  looks — every chart type, the thinking trace, intent/role badges, report +
//  email artifacts, and voice playback. The "voice" plays the reference
//  recordings in v6/audio/ (copied to public/demo/) matched to each answer's
//  language — a stand-in for the cloned XTTS voice the real backend streams.
//
//  Chart specs below mirror exactly what v6/chartspec.py emits (raw value +
//  frozen `<col>_fmt` string), so the demo renders through the real ChartView.
// ============================================================================

import { Artifact, ChartSpec } from "@/lib/api";

export type Lang = "en" | "fr";

export interface DemoTurn {
  user: string;
  intent: string;
  polishRole?: string;
  thinking: string[];
  answer: string;
  lang: Lang;
  chart?: ChartSpec;
  artifacts?: Artifact[];
  /** Optional: override which clip plays. If absent the index-based rotation is used. */
  audioClip?: string;
}

// All 4 reference recordings, in a deterministic rotation.
// We assign them by turn index so every message in the demo plays a different clip.
const ALL_CLIPS = [
  "/demo/English-Femal.mp3",
  "/demo/French-Femal.mp3",
  "/demo/English-Male.mp3",
  "/demo/French-Male.mp3",
];

/** Pick a reference clip for turn `idx`, rotating through all 4 voices. */
export function demoVoiceSrc(idx: number): string {
  return ALL_CLIPS[idx % ALL_CLIPS.length];
}

export const DEMO_TURNS: DemoTurn[] = [
  // 1 — greeting (warm chat persona, French)
  {
    user: "Salam ! Qu'est-ce que tu sais faire exactement ?",
    intent: "greeting",
    polishRole: "chat",
    thinking: ["Let me answer that directly."],
    answer:
      "Salam ! Je suis l'assistant analytique de Djezzy. Je peux interroger la base sur le revenu, l'ARPU, le churn, les abonnés, l'EBITDA, l'OPEX ou le CAPEX — par wilaya et par période — et te sortir un graphique, un email ou un rapport. Par quoi veux-tu commencer ?",
    lang: "fr",
  },

  // 2 — definition (polish)
  {
    user: "C'est quoi exactement l'ARPU ?",
    intent: "definition",
    polishRole: "polish",
    thinking: ["Let me check the reference knowledge first."],
    answer:
      "L'ARPU (Average Revenue Per User) est le revenu moyen généré par abonnée sur une période : le revenu total divisé par le nombre d'abonnés actifs. C'est un indicateur clé pour suivre la valeur d'un parc, indépendamment de sa taille.",
    lang: "fr",
  },

  // 3 — ranking → horizontal bar
  {
    user: "Top 5 wilayas by net revenue in 2025",
    intent: "data",
    polishRole: "analyze",
    thinking: [
      "I'll query the database for the numbers.",
      "Mapped it to tables: global_revenue, dim_location.",
      "Built the query: SELECT wilaya, SUM(net_revenue) … ORDER BY net_revenue DESC LIMIT 5",
      "Ran it — 5 rows back.",
      "The data's in — let me turn it into a chart.",
    ],
    answer:
      "Alger leads net revenue in 2025 at 253.4 million DZD, ahead of Oran (198.1 million DZD) and Constantine (142 million DZD). Annaba and Sétif round out the top five at 118.5 million and 97.3 million DZD. The capital's lead over Oran is about 28%.",
    lang: "en",
    chart: {
      type: "hbar",
      title: "Top 5 wilayas by net revenue (2025)",
      x: { key: "wilaya", label: "Wilaya" },
      series: [{ key: "net_revenue", label: "Net Revenue", unit: "DZD" }],
      data: [
        { wilaya: "Alger", net_revenue: 253387711.02, net_revenue_fmt: "253.4 million DZD" },
        { wilaya: "Oran", net_revenue: 198120044.5, net_revenue_fmt: "198.1 million DZD" },
        { wilaya: "Constantine", net_revenue: 142002233.0, net_revenue_fmt: "142 million DZD" },
        { wilaya: "Annaba", net_revenue: 118540998.0, net_revenue_fmt: "118.5 million DZD" },
        { wilaya: "Sétif", net_revenue: 97331200.0, net_revenue_fmt: "97.3 million DZD" },
      ],
      meta: { rows: 5, kind: "ranking", direction: "desc", lang: "en" },
    },
  },

  // 4 — trend → multi-series line
  {
    user: "Plot the net income trend for Alger and Oran over 2025",
    intent: "data",
    polishRole: "analyze",
    thinking: [
      "I'll query the database for the numbers.",
      "Resolved wilayas: Alger, Oran.",
      "Built the query: SELECT month_start, wilaya, net_income … ORDER BY month_start",
      "Ran it — 12 rows back.",
      "The data's in — let me turn it into a chart.",
    ],
    answer:
      "Both cities trend upward through the first half of 2025. Alger climbs from 12 million DZD in January to 17 million DZD in June, while Oran rises more gently from 9 million to 13.4 million DZD. The gap between them widens over the period.",
    lang: "en",
    chart: {
      type: "line",
      title: "Net income trend — Alger vs Oran (2025)",
      x: { key: "month_start", label: "Month Start" },
      series: [
        { key: "Alger", label: "Alger", unit: "DZD" },
        { key: "Oran", label: "Oran", unit: "DZD" },
      ],
      data: [
        { month_start: "2025-01", Alger: 12000000, Alger_fmt: "12 million DZD", Oran: 9000000, Oran_fmt: "9 million DZD" },
        { month_start: "2025-02", Alger: 13500000, Alger_fmt: "13.5 million DZD", Oran: 9800000, Oran_fmt: "9.8 million DZD" },
        { month_start: "2025-03", Alger: 14200000, Alger_fmt: "14.2 million DZD", Oran: 10500000, Oran_fmt: "10.5 million DZD" },
        { month_start: "2025-04", Alger: 15100000, Alger_fmt: "15.1 million DZD", Oran: 11400000, Oran_fmt: "11.4 million DZD" },
        { month_start: "2025-05", Alger: 16050000, Alger_fmt: "16.1 million DZD", Oran: 12300000, Oran_fmt: "12.3 million DZD" },
        { month_start: "2025-06", Alger: 17000000, Alger_fmt: "17 million DZD", Oran: 13400000, Oran_fmt: "13.4 million DZD" },
      ],
      meta: { rows: 12, kind: "trend", category: "Wilaya", lang: "en" },
    },
  },

  // 5 — comparison → vertical bar
  {
    user: "Compare total OPEX between marketing and distribution in H2 2025",
    intent: "data",
    polishRole: "analyze",
    thinking: [
      "I'll query the database for the numbers.",
      "Mapped it to tables: opex_capex.",
      "Built the query: SELECT cost_center, SUM(opex) … GROUP BY cost_center",
      "Ran it — 2 rows back.",
    ],
    answer:
      "Marketing OPEX reached 54.2 million DZD in the second half of 2025, about 39% higher than distribution at 38.9 million DZD.",
    lang: "en",
    chart: {
      type: "bar",
      title: "OPEX — marketing vs distribution (H2 2025)",
      x: { key: "cost_center", label: "Cost Center" },
      series: [{ key: "opex", label: "OPEX", unit: "DZD" }],
      data: [
        { cost_center: "Marketing", opex: 54200000, opex_fmt: "54.2 million DZD" },
        { cost_center: "Distribution", opex: 38900000, opex_fmt: "38.9 million DZD" },
      ],
      meta: { rows: 2, kind: "comparison", lang: "en" },
    },
  },

  // 6 — composition → donut
  {
    user: "What's the revenue share by segment?",
    intent: "data",
    polishRole: "analyze",
    thinking: [
      "I'll query the database for the numbers.",
      "Built the query: SELECT segment, SUM(revenue) … GROUP BY segment",
      "Ran it — 2 rows back.",
    ],
    answer:
      "Prepaid drives the majority of revenue at 612 million DZD (about 61%), with postpaid contributing 388 million DZD (about 39%).",
    lang: "en",
    chart: {
      type: "pie",
      title: "Revenue share by segment",
      x: { key: "segment", label: "Segment" },
      series: [{ key: "revenue", label: "Revenue", unit: "DZD" }],
      data: [
        { segment: "Prepaid", revenue: 612000000, revenue_fmt: "612 million DZD" },
        { segment: "Postpaid", revenue: 388000000, revenue_fmt: "388 million DZD" },
      ],
      meta: { rows: 2, kind: "composition", lang: "en" },
    },
  },

  // 7 — single-series trend → area (a percentage KPI)
  {
    user: "Show the monthly postpaid churn rate for Alger in 2025",
    intent: "data",
    polishRole: "analyze",
    thinking: [
      "I'll query the database for the numbers.",
      "Built the query: SELECT month_start, churn_rate … ORDER BY month_start",
      "Ran it — 6 rows back.",
    ],
    answer:
      "Postpaid churn in Alger stayed low and stable, easing from 1.42% in January to 1.18% in June — a healthy downward drift over the half-year.",
    lang: "en",
    chart: {
      type: "area",
      title: "Postpaid churn rate — Alger (2025)",
      x: { key: "month_start", label: "Month Start" },
      series: [{ key: "churn_rate", label: "Churn Rate", unit: "%" }],
      data: [
        { month_start: "2025-01", churn_rate: 1.42, churn_rate_fmt: "1.42%" },
        { month_start: "2025-02", churn_rate: 1.36, churn_rate_fmt: "1.36%" },
        { month_start: "2025-03", churn_rate: 1.31, churn_rate_fmt: "1.31%" },
        { month_start: "2025-04", churn_rate: 1.27, churn_rate_fmt: "1.27%" },
        { month_start: "2025-05", churn_rate: 1.22, churn_rate_fmt: "1.22%" },
        { month_start: "2025-06", churn_rate: 1.18, churn_rate_fmt: "1.18%" },
      ],
      meta: { rows: 6, kind: "trend", lang: "en" },
    },
  },

  // 8 — single KPI → stat card, plus an email draft artifact
  {
    user: "Total prepaid ARPU for Oran last month, and email it to the CEO",
    intent: "data",
    polishRole: "analyze",
    thinking: [
      "I'll query the database for the numbers.",
      "Built the query: SELECT AVG(arpu) … WHERE segment='prepaid'",
      "Ran it — 1 row back.",
      "Let me draft an email with these results.",
      "Drafted an email to the CEO.",
    ],
    answer:
      "Prepaid ARPU for Oran last month was 312.45 DZD. I've drafted an email to the CEO with the figure — review it before sending.",
    lang: "en",
    chart: {
      type: "stat",
      title: "Prepaid ARPU — Oran (last month)",
      series: [{ key: "arpu", label: "Prepaid ARPU", unit: "DZD" }],
      data: [{ arpu: 312.45, arpu_fmt: "312.45 DZD" }],
      meta: { rows: 1, kind: "single", lang: "en" },
    },
    artifacts: [
      {
        kind: "email",
        to: "ceo@djezzy.dz",
        to_name: "the CEO",
        subject: "Prepaid ARPU — Oran (last month)",
        body: `Dear CEO,

Please find below the latest prepaid ARPU figure for the Oran region:

  • Period   : May 2026 (last full month)
  • Segment  : Prepaid
  • Wilaya   : Oran
  • ARPU     : 312.45 DZD

This figure is in line with the national prepaid trend and represents a
+2.1% improvement versus the prior month.

Best regards,
Djezzy AI Analytics`,
      },
    ],
  },

  // 9 — cross-turn report (French) — reuses the ARPU numbers from turn 8
  {
    user: "Mets la dernière analyse dans un rapport",
    intent: "data",
    thinking: ["Let me put this into a report.", "Report saved."],
    answer: "Rapport généré à partir du résultat précédent.",
    lang: "fr",
    artifacts: [
      {
        kind: "report",
        name: "report_20260530_141500.md",
        reportSpec: {
          title: "ARPU Prépayé — Oran (Mai 2026)",
          subtitle: "Rapport analytique Djezzy",
          generated_at: "2026-05-30T14:15:00",
          sections: [
            {
              type: "metrics" as const,
              items: [
                { label: "ARPU Prépayé — Oran", value: "312.45 DZD", change: "+2.1 % vs M-1", trend: "up" as const },
                { label: "Segment", value: "Prépayé" },
                { label: "Wilaya", value: "Oran" },
                { label: "Période", value: "Mai 2026" },
              ],
            },
            { type: "heading" as const, text: "Résumé exécutif" },
            {
              type: "text" as const,
              text: "L'ARPU prépayé pour la wilaya d'Oran s'établit à 312,45 DZD au mois de mai 2026, en hausse de 2,1 % par rapport au mois précédent. Cette performance est conforme à la tendance nationale du segment prépayé et confirme la solidité du parc d'abonnés dans cette région.",
            },
            {
              type: "chart" as const,
              spec: {
                type: "stat" as const,
                title: "ARPU Prépayé — Oran (dernière période)",
                series: [{ key: "arpu", label: "ARPU Prépayé", unit: "DZD" }],
                data: [{ arpu: 312.45, arpu_fmt: "312.45 DZD" }],
                meta: { rows: 1, kind: "single", lang: "fr" },
              },
            },
            { type: "heading" as const, text: "Données comparatives" },
            {
              type: "table" as const,
              columns: ["Indicateur", "Valeur", "Variation"],
              rows: [
                ["ARPU Prépayé — Oran (Mai 2026)",   "312.45 DZD", "+2.1 %"],
                ["ARPU Prépayé — Oran (Avr 2026)",   "305.98 DZD", "référence"],
                ["ARPU Prépayé — National (Mai 2026)", "298.70 DZD", "+1.4 %"],
                ["ARPU Postpayé — Oran (Mai 2026)",  "487.20 DZD", "+0.9 %"],
              ],
            },
          ],
        },
      },
    ],
  },

  // 10 — off-topic deflection (verbatim, never polished)
  {
    user: "Can you write me a Python function to sort a list?",
    intent: "off_topic",
    polishRole: "verbatim",
    thinking: ["Let me answer that directly."],
    answer:
      "That's outside what I do — I'm the Djezzy telecom analytics assistant. I can help with KPIs like revenue, ARPU, churn, subscribers, EBITDA, OPEX or CAPEX for any wilaya or period, and I can chart, email, or report the results.",
    lang: "en",
  },
];
