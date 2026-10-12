# DESIGN_SYSTEM.md

Applies to `dashboard/` only (Kiran, Person C). Defines the exact visual tokens so the dashboard is consistent without needing design-by-committee during Phase 5.

## Color palette
Desaturated, clinical, calm — deliberately not a generic SaaS-purple-gradient or a flashy fintech neon look. Amber is reserved exclusively for alert/drift states so its appearance is always meaningful.

| Role | Color | Hex (approx) | Use |
|---|---|---|---|
| Primary | Desaturated blue | `#378ADD` / `#0C447C` (dark text) | Active nodes, primary chart lines, buttons |
| Secondary | Teal | `#1D9E75` / `#085041` (dark text) | Training-in-progress state, secondary chart lines |
| Neutral | Gray | `#888780` / `#444441` (dark text) | Idle state, structural UI, borders |
| Alert (reserved) | Amber | `#EF9F27` / `#854F0B` (dark text) | Drift detected, warnings only — never decorative |
| Danger | Red | `#E24B4A` / `#A32D2D` (dark text) | High-risk inference result, hard failures only |
| Background | White / near-white | `--surface-2` | Card backgrounds |
| Page background | Light gray | `--surface-0` | Page canvas |

Rule: no gradients, no drop shadows beyond functional focus rings, no decorative color — every color used must encode a real state.

## Typography
- Font: Inter or IBM Plex Sans (IBM Plex has stronger "data/health tech" association — prefer it if choosing between the two)
- Headings: 22px/18px/16px, weight 500 only
- Body: 16px, weight 400, line-height 1.7
- Numeric/metric values (accuracy %, epsilon, confidence scores): monospace or tabular-figure variant for visual precision — e.g. `font-variant-numeric: tabular-nums`
- Sentence case everywhere — never Title Case, never ALL CAPS, including chart labels and panel headings
- No text below 11px anywhere

## Layout
- Dashboard-grid, not a scrolling narrative page — system state should be visible at a glance without scrolling on a standard laptop screen during a live demo
- Card-based panels: white background, 0.5px border, 12px corner radius, ~1rem padding
- Metric cards (single numbers like accuracy, epsilon, F1): muted 13px label above, 24px/500 weight number below

## Components
- **Buttons**: transparent background, 0.5px border, hover state fills lightly — the "Inject Drift" button is the one exception that may use the amber/alert color to signal its consequence, since it's the one destructive/triggering action in the UI
- **Status chips** (client states): small pill, colored background at the light tint of the role color, colored text at the dark tint of the same ramp — never black text on a colored chip
- **Charts**: Chart.js or Recharts, flat fills, no 3D effects, always labeled axes with units, legend required whenever more than one series is shown, error bars/variance shown wherever multiple seeds exist (Phase 7+ plots)

## Icons
Use a consistent outline icon set (e.g. Tabler icons) throughout — server/device icon for simulated clients, shield icon for privacy panel, alert-triangle for drift, chart icons for training monitor. No emoji anywhere in the UI.

## Motion
Subtle only — chart lines animating in on data update, smooth transitions on status chip color changes. No decorative animation, no bouncing/pulsing effects except a clear, purposeful "training in progress" indicator (e.g. a slow pulse on the teal status chip only).

## Accessibility baseline
- Color is never the only signal — status chips carry text labels alongside color, not color alone
- Sufficient contrast between text and background at every color-role pairing listed above (use the darkest matching stop for text on any colored background)
- All interactive elements (Inject Drift button, form inputs) keyboard-accessible

## What this design system explicitly rejects
No dark-mode-by-default neon/glassmorphism aesthetic (as seen in general fintech/crypto landing page references) — that register reads as visually mismatched with a clinical decision-support tool and undermines the project's credibility with evaluators. Confidence in this UI comes from clarity, restraint, and correct data — not visual flourish.
