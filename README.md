# Data Centre Optimization and Flexibility Analysis

This repository contains a comprehensive data center (DC) optimization framework that models and analyzes the operational flexibility of data centers, focusing on cost minimization, thermal management, and demand response capabilities.

## Overview

The codebase implements a mixed-integer linear programming (MILP) optimization model for data center operations using Pyomo. It simulates various aspects of data center operations including IT workload scheduling, cooling systems, energy storage (UPS and TES), and electricity cost optimization.

## Key Features

- **Multi-objective optimization**: Minimizes electricity costs while maintaining operational constraints
- **Flexible workload scheduling**: Models shiftable IT workloads with different delay tolerances
- **Advanced cooling system**: Includes thermal energy storage (TES) and HVAC systems
- **Energy storage modeling**: UPS battery and TES systems for energy arbitrage
- **Real-time pricing**: Incorporates time-varying electricity tariffs
- **Flexibility analysis**: Determines maximum achievable demand response durations and magnitudes

## Project Structure

### Core Files

#### `nominal_calculation.py`
- **Purpose**: Runs the baseline (nominal) case simulation
- **Key Functionality**:
  - Models data center operation without optimization (inflexible workload)
  - Calculates baseline energy consumption and costs
  - Generates comprehensive performance metrics
  - Includes piecewise linear approximation for CPU-power relationship
- **Output**: CSV files with nominal case results, charts showing power consumption, temperatures, and costs

#### `optimisation.py`
- **Purpose**: Main optimization engine for cost-minimizing data center operation
- **Key Functionality**:
  - Flexible IT workload scheduling across time slots
  - Joint optimization of IT, cooling, and energy storage systems
  - Cost minimization subject to operational constraints
  - Post-processing for detailed results analysis
- **Output**: Optimized schedules, cost savings analysis, workload shifting patterns

#### `flexibility_duration.py`
- **Purpose**: Analyzes data center demand response capabilities
- **Key Functionality**:
  - Determines maximum duration for given power reduction targets
  - Tests various flexibility magnitudes and starting times
  - Uses binary search to find optimal durations
  - Generates flexibility heatmaps and detailed results
- **Output**: Flexibility duration curves, grid impact analysis, source-wise power breakdowns

#### `constraints.py`
- **Purpose**: Defines all optimization constraints
- **Key Components**:
  - IT job scheduling constraints with shiftability limits
  - UPS battery energy balance and charging constraints
  - Cooling system thermal dynamics
  - Power balance equations
  - Piecewise linear approximations for non-linear relationships

### Input and Parameters

#### `inputs/parameters_optimisation.py`
- **Purpose**: Central parameter management
- **Key Parameters**:
  - **Time horizons**: 24-hour simulation with 15-minute time steps
  - **IT Equipment**: 100 racks, 10 servers each, power ranges (167-1000 kW)
  - **Energy Storage**:
    - UPS: 600 kWh capacity, 40-2700 kW power range
    - TES: 1000 kWh thermal storage, 300 kW charge/discharge
  - **Cooling System**: COP=5, thermal properties, temperature limits
  - **Electricity Pricing**: Time-of-use tariff structure

### Plotting and Analysis

#### `plotting_and_saving/nom_opt_charts.py`
- **Purpose**: Visualization for nominal case results
- **Charts Generated**:
  - Cost vs energy price over time
  - TES energy levels and heat flows
  - Data center temperature profiles
  - Cooling power consumption breakdown
  - Thermal cooling power by source
  - Cumulative cost analysis

#### `plotting_and_saving/flexibility_duration_results_and_plots.py`
- **Purpose**: Advanced plotting for flexibility analysis
- **Key Features**:
  - Grid plots showing power changes by source
  - Stacked bar charts for energy breakdowns
  - Heatmaps of achievable flexibility durations
  - Source-wise contribution analysis

### Data Management

#### `static/data/` Directory Structure:
- `inputs/`: Load profiles, shiftability matrices, tariff data
- `nominal_outputs/`: Baseline simulation results
- `optimisation_outputs/`: Optimized schedules and load profiles
- `flexibility_outputs/`: Demand response capability analysis

#### Key Data Files:
- `load_profiles.csv`: Inflexible and flexible CPU load profiles
- `shiftability_profile.csv`: Workload shiftability matrix (4 tranches × 96 time slots)
- Various result CSV files with detailed time-series data

### Utility Modules

#### `dc_utilisation/data_centre_utilisation.py`
- **Purpose**: Analysis of real data center utilization patterns
- **Functionality**:
  - Processes anonymized data center demand profiles
  - Identifies most and least variable utilization patterns
  - Generates visualization of utilization variability

## Mathematical Model

### Objective Function
Minimize total electricity cost over the optimization horizon:

```
min ∑(P_grid_it + P_cooling_hvac + P_cooling_tes + P_overhead + P_ups_charge) × price × Δt
```

### Key Constraints

1. **Job Completion**: Flexible workloads must be executed within shiftability windows
2. **CPU-Power Relationship**: Piecewise linear approximation of P = f(CPU)^1.32
3. **Energy Balance**: UPS and TES state transitions
4. **Thermal Dynamics**: IT equipment, rack, and aisle temperature evolution
5. **Power Balance**: Supply-demand equilibrium at each time step
6. **Operational Limits**: Temperature bounds, power limits, storage constraints

### Variables
- **Decision Variables**: Workload scheduling (u_tks), power flows, storage levels, temperatures
- **State Variables**: Energy storage levels, thermal states
- **Derived Variables**: Total power consumption, operating costs

## Usage

### Running Nominal Case
```bash
python nominal_calculation.py
```
Generates baseline performance without optimization.

### Running Optimization
```bash
python optimisation.py
```
Performs cost-minimizing optimization with flexible workloads.

### Flexibility Analysis
```bash
python flexibility_duration.py
```
Analyzes demand response capabilities across different scenarios.

## Dependencies

- **Pyomo**: Optimization modeling framework
- **SCIP**: Mixed-integer solver (recommended)
- **pandas, numpy**: Data manipulation
- **matplotlib, seaborn**: Visualization
- **pathlib**: File system operations

## Key Results and Insights

1. **Cost Savings**: Optimization typically achieves 10-20% cost reduction through load shifting
2. **Flexibility Potential**: Data centers can provide significant demand response (100-500 kW for hours)
3. **Energy Storage Synergy**: Combined UPS and TES operation enables extended flexibility
4. **Thermal Constraints**: Cooling system dynamics limit maximum flexibility durations
5. **Time-of-Use Impact**: Peak pricing drives workload shifting to off-peak periods

## Applications

- **Demand Response**: Quantify data center contribution to grid flexibility
- **Energy Cost Management**: Optimize operations under time-varying tariffs
- **Infrastructure Planning**: Size energy storage and cooling systems
- **Grid Integration**: Assess data center role in renewable energy integration

## Future Extensions

- Multi-day optimization horizons
- Renewable energy integration
- Advanced cooling technologies
- Real-time optimization capabilities
- Machine learning for load prediction
