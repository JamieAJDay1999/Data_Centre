# A1 Poster Plan: *Data Centre Flexibility for Power System Support*

## Summary
Create a portrait A1 academic poster (594 mm × 841 mm) for a mixed technical audience that distills your paper into a clear, high-impact story in ~500 words, centered on one core message: **an integrated IT–UPS–cooling model can both reduce operating cost and provide duration-certified flexibility to the grid**.  
The poster will use 5 figure blocks (from your extracted figures) and emphasize quantified outcomes: **10.02% cost reduction** and strong time-dependent flexibility asymmetry.

## Final Specification

### 1) Canvas, Grid, and Reading Flow
| Item | Specification |
|---|---|
| Size | A1 portrait, 594 × 841 mm |
| Margins | 22 mm outer margin on all sides |
| Grid | 12-column grid, 8 mm gutters, baseline spacing 6 mm |
| Reading path | Top-to-bottom in 4 bands: Context → Method → Results → Impact |
| White space target | 28–35% of poster area |
| Body word target | 480–520 words (excluding title/authors/captions/references) |

### 2) Content Architecture and Word Budget
| Section | Target words | Content to include |
|---|---:|---|
| Problem and Motivation | 70 | DC demand growth and grid strain; flexibility need; why DCs are both challenge and opportunity |
| Research Objective | 45 | Two contributions: least-cost baseline optimization + duration-aware flexibility quantification |
| Integrated Model | 110 | IT shifting, UPS-ESS dispatch, cooling/TES thermodynamics; key assumptions (1 MW IT, 15-min slots, 24h + 3h extension) |
| Scenario Design | 70 | Scenario 1 base, Scenario 2 cost minimization, Scenario 3 duration-aware flexibility (ΔP, t0, τ with recovery window) |
| Key Results | 155 | Cost reduction and temporal/asymmetric flexibility findings with concrete numbers |
| Practical Implications | 50 | What operators/system planners can do now; market relevance |
| Limitations and Next Work | 20 | Pre-conditioning for downward IT flexibility; portfolio aggregation |

Total: **520 words max target**, aim for ~500 in final draft.

### 3) Figure Plan (5 Core Blocks)
| Block | Source file(s) | Placement | Purpose |
|---|---|---|---|
| F1 System Architecture | `extracted_figures/p04_img01.png` | Upper-left (Band 1/2) | Explain integrated physical/electrical/thermal coupling |
| F2 Cost Optimization Signal-Response | `extracted_figures/p12_img01.png` | Mid-left (Band 3) | Show baseline vs optimized power under price signal |
| F3 Asset Dispatch Decomposition | `extracted_figures/p12_img03.png` | Mid-right (Band 3) | Show IT, CRAC, UPS, TES co-optimization behavior |
| F4 Flexibility Envelope Heatmap | `extracted_figures/p13_img01.png` | Center-large (Band 3/4 focal visual) | Show τ across start time and ±ΔP; main novelty |
| F5 Asymmetry Evidence Panel | `extracted_figures/p13_img02.png` + `extracted_figures/p14_img01.png` | Bottom row side-by-side | Compare upward vs downward component contributions |

### 4) Mandatory Quantitative Callouts
Use large numeric callout boxes near F2/F4 with these exact values:
1. Base vs optimized cost: **£1,659.54 → £1,493.19**
2. Absolute saving: **£166.34**
3. Relative saving: **10.02%**
4. Flexibility example: **-100 kW upward flexibility for 6.8 h at 00:15 vs 0.2 h at 17:30**
5. Model scope: **1 MW IT capacity, UPS 600 kWh, TES 1000 kWh**

### 5) Visual Design System
| Element | Spec |
|---|---|
| Visual direction | Clean technical editorial; warm light background with high-contrast data accents |
| Background | #F5F4EF (off-white) |
| Primary text | #102A43 |
| Accent 1 (upward flexibility) | #117A65 |
| Accent 2 (downward flexibility) | #D97706 |
| Neutral lines | #6B7280 |
| Heading font | `Montserrat` Semibold |
| Body font | `Source Sans 3` Regular |
| Numeric/callouts | `IBM Plex Mono` Semibold |
| Title size | 90–105 pt |
| Section headers | 44–54 pt |
| Body text | 26–30 pt |
| Captions | 20–22 pt |

### 6) Poster Structure (Decision-Complete Layout)
1. Top band (15% height): Title, authors, affiliation, 1-sentence thesis.
2. Band 2 (22%): Left F1; right “Objective + Method at a glance” text block.
3. Band 3 (33%): F2 left and F3 right with short interpretation text under each.
4. Band 4 (22%): F4 full-width as the visual centerpiece with two callout boxes.
5. Bottom band (8%): F5 split panel + implications + 3-item future work + QR/reference strip.

### 7) Public Interfaces / Deliverables
1. Final print file: `poster_A1_data_centre_flexibility.pdf` (CMYK, 300 dpi, embedded fonts).
2. Editable source: `poster_A1_data_centre_flexibility.pptx` or `.ai` (choose one and keep linked assets local).
3. Asset manifest: one-page text listing each figure source path and caption.

### 8) Test Cases and Acceptance Criteria
1. Word count test: body text between 480 and 520 words.
2. Data integrity test: all numeric claims match paper values exactly.
3. Figure legibility test: all axis labels readable from 1.2 m viewing distance.
4. Narrative test: reader can answer “what is new, what is achieved, why it matters” in under 90 seconds.
5. Print preflight test: no rasterized text, no missing fonts, no low-res images at placed size.
6. Asymmetry comprehension test: upward vs downward flexibility difference is explicitly visible in F4/F5 and stated in text.

### 9) Implementation Sequence
1. Place fixed canvas/grid and define typography/color tokens.
2. Import and crop figures to the exact 5-block layout.
3. Draft ~500-word copy to section word budgets.
4. Add mandatory quantitative callouts.
5. Tighten captions to one insight sentence each.
6. Run acceptance tests and preflight for print/export.

## Assumptions and Defaults
1. Orientation locked to **A1 portrait**.
2. Audience locked to **mixed technical**.
3. Figure density locked to **5 core figure blocks**.
4. No extra equations beyond minimal notation for ΔP, t0, τ.
5. No additional Codex skill is required for this task (available skills are for skill creation/installation, not poster planning).
