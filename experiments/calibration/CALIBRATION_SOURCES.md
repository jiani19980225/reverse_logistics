# Parameter Calibration Sources

All parameters are derived from public data BEFORE running the simulation.
The simulation output is whatever it is — we do not adjust parameters to
match a target.

## S1: IT Infrastructure

### Component Prices (USD, used/refurbished market)

Source: PCPartPicker used market prices, eBay completed listings (2023-2024),
IT asset disposition industry reports.

| Component | New price | Used recovery value | Source |
|-----------|-----------|-------------------|--------|
| CPU (Xeon E5/Gold) | $800-2000 | $150-400 | eBay completed listings for Xeon E5-2680v4, Gold 6130 |
| DIMM (16GB DDR4 ECC) | $40-80 new | $15-30 used | PCPartPicker DDR4 ECC pricing |
| GPU (Tesla/A100) | $5000-15000 | $800-3000 | eBay data center GPU listings |
| SSD (enterprise 1TB) | $150-300 | $40-100 | PCPartPicker enterprise SSD |
| HDD (enterprise 4TB) | $80-150 | $20-50 | Backblaze drive cost reports |

**Decision**: Use midpoint of used recovery value range.
- CPU: $275
- DIMM: $22
- GPU: $1900
- SSD: $70
- HDD: $35

### Yield Rates by Age

Source: Backblaze Annual Drive Stats (2023), WEEE recovery statistics (Eurostat),
industry decommission reports.

| Component | 0-3 years | 3-5 years | Source |
|-----------|-----------|-----------|--------|
| CPU | 92-98% | 78-88% | Very low failure rate per Intel reliability data |
| DIMM | 88-96% | 72-85% | Google/Facebook memory error studies |
| GPU | 85-95% | 70-85% | Data center GPU lifecycle reports |
| SSD | 82-92% | 45-65% | Backblaze SSD stats (P/E cycle cliff) |
| HDD | 72-85% | 32-55% | Backblaze AFR by age (1.5% yr1 → 5-8% yr5) |

**Decision**: Use midpoint of each range as base_yield (weighted 60% young, 40% old).
- CPU: 0.60*0.95 + 0.40*0.83 = 0.902
- DIMM: 0.60*0.92 + 0.40*0.79 = 0.868
- GPU: 0.60*0.90 + 0.40*0.78 = 0.852
- SSD: 0.60*0.87 + 0.40*0.55 = 0.742
- HDD: 0.60*0.79 + 0.40*0.44 = 0.650

### Disassembly Costs

Source: ITAD industry estimates, loaded labor rate $60-90/hr (BLS data for
electronics recycling technicians, NAICS 562920).

| Level | Time | Cost at $75/hr | Source |
|-------|------|---------------|--------|
| L0→L1 (open chassis, remove trays) | 8-12 min | $10-15 | iFixit server teardown times |
| L1→L2 (extract socketed components) | 25-40 min | $31-50 | Industry estimates |
| L1 inspection (functional test) | 10-20 min | $12-25 | POST test + visual |
| L2 inspection (component test) | 45-75 min | $56-94 | Individual component bench test |

**Decision**: Use midpoints.
- L0→L1: $13, 10 min
- L1→L2: $40, 32 min
- L1 inspection: $19, 15 min
- L2 inspection: $75, 60 min

### Capacity

Source: Typical ITAD facility processes 50-200 servers/day with 10-20 technicians.
At 8hr/day, 15 techs: 15*8*5 = 600 labor-hours/week.

**Decision**: 600 hours/week for 500 assets.

### Note Distribution

Source: Industry decommission statistics (Sims Lifecycle Services annual report,
Iron Mountain ITAD white papers).

| Condition | Fraction | Source |
|-----------|----------|--------|
| Clean/routine | 25% | Planned refresh cycles |
| Mixed signals | 35% | Partial issues found during decom |
| Damaged | 25% | Failed in service |
| Ambiguous | 15% | Incomplete documentation |

---

## S2: Aviation MRO

### Component Prices (USD, serviceable/overhauled)

Source: Aviation parts market (ILS, PartBase, HEICO pricing), FAA repair
station rate surveys.

| Component | Serviceable value | Source |
|-----------|------------------|--------|
| Skin panel | $2,000-5,000 | Airframe structural repair estimates |
| Frame section | $3,000-8,000 | Major structural component |
| Battery (aircraft) | $800-2,000 | Aircraft battery replacement cost |
| Lighting unit | $200-500 | Cabin lighting assemblies |
| Seat assembly | $1,500-4,000 | Airline seat refurbishment market |
| Door assembly | $5,000-15,000 | Complex mechanical assembly |
| Flight surface | $8,000-20,000 | Control surface overhaul |
| Turbine blade | $2,000-6,000 | Per-blade overhaul value |
| Bearing | $400-1,200 | Engine/APU bearings |
| Nav unit | $8,000-25,000 | Avionics navigation equipment |
| Comm radio | $3,000-8,000 | Communication equipment |
| Display | $2,000-5,000 | Cockpit/cabin displays |

**Decision**: Use midpoint of serviceable value.
- skin_panel: 3500, frame_section: 5500, battery: 1400, lighting_unit: 350
- seat_assembly: 2750, door_assembly: 10000, flight_surface: 14000
- turbine_blade: 4000, bearing: 800, nav_unit: 16500, comm_radio: 5500, display: 3500

### Yield (from condition)

Source: FAA SDR condition distribution mapped to recovery probability.
MRO industry: ~60-70% of removed components are returned to serviceable.

| Condition | Recovery probability | Source |
|-----------|---------------------|--------|
| Serviceable | 85-95% | Already serviceable, minor paperwork |
| Worn | 50-70% | May be overhauled |
| Corroded | 25-50% | Depends on severity |
| Cracked | 15-35% | Often beyond economic repair |
| Inoperative | 35-55% | May be repairable |
| Failed | 5-20% | Usually scrap |

Weighted average yield (using SDR distribution):
0.28*0.90 + 0.25*0.60 + 0.20*0.38 + 0.17*0.25 + 0.07*0.45 + 0.03*0.13 = 0.556

**Decision**: base_yield per component = 0.56 (uniform, adjusted by condition in simulation).

### Inspection/Processing Costs

Source: FAA Part 145 repair station hourly rates ($85-150/hr), typical
inspection times from MRO industry.

- L0 (unit exchange): $100, 30 min
- L1 (component overhaul): $600, 150 min
- Inspection: $100, 35 min (includes documentation per Part 145)

### Capacity

Source: Mid-size MRO shop, 20 mechanics, 8hr/day, 5 days.
20*8*5 = 800 hours/week.

**Decision**: 800 hours/week for 500 assets.

---

## S3: Consumer Electronics

### Component Prices (USD, material recovery + refurb value)

Source: iFixit parts pricing, material recovery values from WEEE literature,
refurbishment market (Back Market, Decluttr pricing).

Key insight: S3 value comes primarily from WHOLE-UNIT refurbishment (resale),
not component-level material recovery. A functional iPhone resells for $150-400;
its material recovery value is only $2-5.

| Product | Refurb resale | Material recovery | Source |
|---------|--------------|-------------------|--------|
| Smartphone | $80-250 | $2-5 | Back Market, Decluttr |
| Laptop | $150-400 | $5-15 | Refurbished market |
| Tablet | $100-300 | $3-8 | Refurbished market |

For simulation, we model per-component values that sum to whole-unit value:
- Functional unit (L0 refurb): ~$120 avg smartphone, ~$250 laptop, ~$180 tablet
- Component harvest (L1): ~$15 smartphone, ~$35 laptop, ~$20 tablet

**Decision** (per-component values representing share of whole-unit value):
- pcb: $8, battery: $5, display: $15, camera: $3, keyboard: $7, ssd: $12

These are LOW because most S3 value is at L0 (whole-unit), not L1 (components).
The simulation models L2 (component-level) processing.

### Yield

Source: CPSC remedy distribution, NRF returns data.
- Refund (functional, resellable): 42% → yield ~0.90
- Repair (fixable): 38% → yield ~0.55
- Replace (defective): 20% → yield ~0.15

Weighted: 0.42*0.90 + 0.38*0.55 + 0.20*0.15 = 0.617

**Decision**: base_yield = 0.62 (uniform across components).

### Costs

Source: Returns processing industry (Optoro, Returnly white papers).
- Inspection: $15-25 per unit, 3-5 min (visual + power-on test)
- Processing L1: $8-15, 5-10 min
- Processing L2: $20-35, 10-15 min
- Rework (discovered bad mid-process): $25-40 (repackaging + disposal)

**Decision**:
- L2 inspection: $20, 5 min
- L1 inspection: $10, 3 min
- L2 processing: $12, 10 min
- L1 processing: $6, 4 min
- Rework: $30, 8 min

### Capacity

Source: Large returns center processes 5,000-20,000 units/day.
For 1000-unit daily batch: 20 workers * 8hr = 160 hours/day.

**Decision**: 160 hours/day.

---

## Extractor Parameters

### Keyword signal vocabularies
Source: iFixit teardown reports (S1), FAA SDR text analysis (S2),
Amazon review defect language (S3). These are documented in the paper.

### Confidence penalties
- S2: σ *= 0.6 (standardized ATA codes reduce keyword effectiveness)
- S3: high σ with random φ (consumer language has near-zero yield correlation)

Source: Paper Section IV-B confidence calibration analysis.

### Thresholds (τ_h, τ_l)
Source: Paper Section V-B sensitivity analysis.
- S1/S2: τ_h=0.50, τ_l=0.25
- S3: τ_h=0.50, τ_l=0.45

---

## What We Do NOT Calibrate to Target

- We do NOT adjust prices to hit a specific TRV
- We do NOT adjust yields to hit a specific lift
- We do NOT adjust capacity to hit a specific DCR
- The simulation output is whatever it is
- If lift is negative, that is a finding (framework doesn't help in that scenario)
- If lift is 2% instead of 8%, we report 2%
