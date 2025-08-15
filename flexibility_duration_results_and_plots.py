import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import pathlib
import seaborn as sns
import pulp

DATA_DIR = pathlib.Path("static/data")
IMAGE_DIR = pathlib.Path("static/images")

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def extract_detailed_results(m, params, data, start_timestep, flex_time, baseline_df):
    """
    MODIFICATION: This function now also extracts temperature data.
    """
    def get_val(var_name):
        var = m.variablesDict().get(var_name)
        return pulp.value(var) if var is not None else 0

    time_slots = list(params.TEXT_SLOTS)
    results_df = pd.DataFrame(index=time_slots)
    results_df.index.name = 'Time_Slot_EXT'

    final_columns = []

    # --- Power Metrics ---
    power_sources = [
        'P_IT_Total_kW', 'P_Grid_IT_kW', 'P_Chiller_HVAC_kW',
        'P_Chiller_TES_kW', 'P_Grid_Cooling_kW', 'P_Grid_Other_kW',
        'P_UPS_Charge_kW', 'P_UPS_Discharge_kW'
    ]

    results_df['P_Total_kw_Opt'] = [
        get_val(f"P_Grid_IT_{s}") +
        (get_val(f"P_Chiller_HVAC_Watts_{s}") / 1000.0) +
        (get_val(f"P_Chiller_TES_Watts_{s}") / 1000.0) +
        get_val(f"P_Grid_Other_{s}") +
        get_val(f"P_UPS_Charge_{s}")
        for s in time_slots
    ]
    results_df['P_Total_kw_Base'] = [baseline_df.loc[s, 'P_Total_kW'] if s in baseline_df.index else 0 for s in time_slots]
    results_df['P_Total_kw_Diff'] = results_df['P_Total_kw_Opt'] - results_df['P_Total_kw_Base']
    final_columns.extend(['P_Total_kw_Base', 'P_Total_kw_Opt', 'P_Total_kw_Diff'])

    for source in power_sources:
        if source in ['P_Chiller_HVAC_kW', 'P_Chiller_TES_kW']:
            var_base_name = source.replace('_kW', '_Watts')
            opt_vals = [get_val(f"{var_base_name}_{s}") / 1000.0 for s in time_slots]
        elif source == 'P_Grid_Cooling_kW':
            hvac = [get_val(f"P_Chiller_HVAC_Watts_{s}") / 1000.0 for s in time_slots]
            tes = [get_val(f"P_Chiller_TES_Watts_{s}") / 1000.0 for s in time_slots]
            opt_vals = [h + t for h, t in zip(hvac, tes)]
        else:
            var_base_name = source.replace('_kW','')
            opt_vals = [get_val(f"{var_base_name}_{s}") for s in time_slots]

        base_vals = [baseline_df.loc[s, source] if s in baseline_df.index else 0 for s in time_slots]
        diff_vals = [opt - base for opt, base in zip(opt_vals, base_vals)]
        base_col, opt_col, diff_col = f'{source}_base', f'{source}_opt', f'{source}_diff'
        results_df[base_col] = base_vals
        results_df[opt_col] = opt_vals
        results_df[diff_col] = diff_vals
        final_columns.extend([base_col, opt_col, diff_col])

    # --- Energy Storage Metrics ---
    e_tes_opt = [get_val(f"E_TES_{s}") for s in time_slots]
    e_tes_base = [baseline_df.loc[s, 'E_TES_kWh'] if s in baseline_df.index and 'E_TES_kWh' in baseline_df.columns else 0 for s in time_slots]
    results_df['E_TES_kWh_base'] = e_tes_base
    results_df['E_TES_kWh_opt'] = e_tes_opt
    results_df['E_TES_kWh_diff'] = [opt - base for opt, base in zip(e_tes_opt, e_tes_base)]
    final_columns.extend(['E_TES_kWh_base', 'E_TES_kWh_opt', 'E_TES_kWh_diff'])

    # --- START: New Temperature Metrics Extraction ---
    temp_sources = {
        'T_IT_Celsius': 'T_IT',
        'T_Rack_Celsius': 'T_Rack',
        'T_ColdAisle_Celsius': 'T_ColdAisle',
        'T_HotAisle_Celsius': 'T_HotAisle'
    }
    
    for base_name, pulp_name in temp_sources.items():
        opt_vals = [get_val(f"{pulp_name}_{s}") for s in time_slots]
        base_vals = [baseline_df.loc[s, base_name] if s in baseline_df.index and base_name in baseline_df.columns else 0 for s in time_slots]
        diff_vals = [opt - base for opt, base in zip(opt_vals, base_vals)]
        
        base_col, opt_col, diff_col = f'{base_name}_base', f'{base_name}_opt', f'{base_name}_diff'
        results_df[base_col] = base_vals
        results_df[opt_col] = opt_vals
        results_df[diff_col] = diff_vals
        final_columns.extend([base_col, opt_col, diff_col])
    # --- END: New Temperature Metrics Extraction ---


    # --- Cost and Nominal Load Metrics ---
    results_df['Price_GBP_per_MWh'] = [data['electricity_price'][s] if s < len(data['electricity_price']) else 0 for s in time_slots]
    results_df['P_IT_Nominal'] = [data['Pt_IT_nom_TEXT'][s] if s < len(data['Pt_IT_nom_TEXT']) else 0 for s in time_slots]
    final_columns.extend(['Price_GBP_per_MWh', 'P_IT_Nominal'])

    # --- CPU Load Metrics ---
    results_df['Inflexible_Load_CPU_Nom'] = [data['inflexibleLoadProfile_TEXT'][s] if s < len(data['inflexibleLoadProfile_TEXT']) else 0 for s in time_slots]
    results_df['Flexible_Load_CPU_Nom'] = [data['flexibleLoadProfile_TEXT'][s] if s < len(data['flexibleLoadProfile_TEXT']) else 0 for s in time_slots]

    total_cpu_load_opt = [get_val(f"TotalCpuUsage_{s}") for s in time_slots]
    total_cpu_load_base = [baseline_df.loc[s, 'Total_CPU_Load'] if s in baseline_df.index else 0 for s in time_slots]

    ut_ks_vars = {k: v.value() for k, v in m.variablesDict().items() if "U_JobTranche" in k and v.value() is not None}
    flexible_cpu_usage = {s: 0 for s in time_slots}
    for var_name, var_value in ut_ks_vars.items():
        try:
            tuple_str = var_name.split("U_JobTranche_(")[1].strip(")")
            parts = [p.strip() for p in tuple_str.split(',')]
            if len(parts) == 3:
                _, _, s_job = map(int, parts)
                if s_job in flexible_cpu_usage:
                    flexible_cpu_usage[s_job] += var_value
        except (IndexError, ValueError):
            continue

    flexible_cpu_usage_list = [flexible_cpu_usage[s] for s in time_slots]
    results_df['Flexible_Load_CPU_Opt'] = flexible_cpu_usage_list
    results_df['Inflexible_Load_CPU_Opt'] = [total - flex for total, flex in zip(total_cpu_load_opt, flexible_cpu_usage_list)]
    results_df['Total_CPU_Load_base'] = total_cpu_load_base
    results_df['Total_CPU_Load_opt'] = total_cpu_load_opt

    final_columns.extend([
        'Inflexible_Load_CPU_Nom', 'Flexible_Load_CPU_Nom',
        'Total_CPU_Load_base', 'Total_CPU_Load_opt',
        'Inflexible_Load_CPU_Opt', 'Flexible_Load_CPU_Opt'
    ])

    results_df = results_df[final_columns]

    return results_df

def plot_grid_flex_contributions(results_df, start_ts, fm, dur_steps):
    """
    Stacked bars of (Optimised - Baseline) grid power by source for the flex window.
    One stack per timestep. Dashed black line at the flex magnitude. Grey 'X' = sum of diffs.
    Nicer colors + legend. 
    """
    # Flex window only
    t0, t1 = start_ts, start_ts + dur_steps   # [t0, t1)
    window_idx = [t for t in results_df.index if t0 <= t < t1]
    if not window_idx:
        print("No timesteps in window to plot.")
        return

    # Diff columns that draw from the grid (defensive: keep only those present)
    col_map = {
        'P_Grid_IT_kW_diff':      'IT (grid)',
        'P_Chiller_HVAC_kW_diff': 'Chiller HVAC',
        'P_Chiller_TES_kW_diff':  'Chiller TES',
        'P_Grid_Other_kW_diff':   'Other overhead',
        'P_UPS_Charge_kW_diff':   'UPS charge',
    }
    cols_present = [c for c in col_map if c in results_df.columns]
    if not cols_present:
        print("No suitable *_diff columns found to plot.")
        return

    diff_df = results_df.loc[window_idx, cols_present].rename(columns=col_map)

    # --- Aesthetics ----------------------------------------------------------
    # Okabe–Ito palette (color-blind friendly)
    palette = {
        'IT (grid)':      '#0072B2',
        'Chiller HVAC':   '#E69F00',
        'Chiller TES':    '#009E73',
        'Other overhead': '#CC79A7',
        'UPS charge':     '#56B4E9',
    }
    series = [c for c in diff_df.columns]  # preserve order

    x = np.arange(len(diff_df))
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    width = 0.82
    edge = 'white'

    pos_bottom = np.zeros(len(diff_df))
    neg_bottom = np.zeros(len(diff_df))

    legend_swatches = []

    # Draw stacked bars, one color per series (same for pos/neg)
    for name in series:
        s = diff_df[name].values
        color = palette.get(name, None)

        s_pos = np.clip(s, 0, None)
        s_neg = np.clip(s, None, 0)

        ax.bar(x, s_pos, width=width, bottom=pos_bottom, color=color, edgecolor=edge, linewidth=0.6)
        pos_bottom = pos_bottom + s_pos

        ax.bar(x, s_neg, width=width, bottom=neg_bottom, color=color, edgecolor=edge, linewidth=0.6)
        neg_bottom = neg_bottom + s_neg

        legend_swatches.append(Patch(facecolor=color, edgecolor=edge, label=name))

    # Grey X at the sum of differences
    sum_diffs = diff_df.sum(axis=1).values
    ax.scatter(x, sum_diffs, marker='x', s=64, color='#666666', linewidths=1.5, label='Sum of diffs')

    # Flex target line (faded black, dashed)
    ax.axhline(fm, color='black', alpha=0.5, linewidth=2, linestyle='--', label=f'Flex target = {fm} kW')

    # Titles, labels
    """ax.set_title(
        f'Grid power change by source (opt - base)\nStart {start_ts}, Flex {fm} kW, Duration {dur_steps} steps',
        pad=14, fontsize=16, weight='bold'
    )"""
    ax.set_xlabel('Time Slot', fontsize=12)
    ax.set_ylabel('Δ Grid Power vs Baseline (kW)', fontsize=12)

    # X ticks (sparse for readability)
    tick_every = max(1, len(x) // 16)
    ax.set_xticks(x[::tick_every])
    ax.set_xticklabels([str(t) for t in window_idx[::tick_every]], rotation=45, ha='right')

    # Light y grid; hide extraneous spines
    ax.grid(axis='y', linestyle=':', color='#E0E0E0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend (series swatches + line + marker), outside the plot area
    extra = [
        Line2D([0], [0], linestyle='--', color='black', alpha=0.5, lw=2, label='Flex target'),
        Line2D([0], [0], marker='x', linestyle='None', color='#666666', markersize=8, label='Sum of diffs'),
    ]
    legend_items = legend_swatches + extra
    legend_labels = [p.get_label() for p in legend_items]
    ax.legend(
        legend_items, legend_labels,
        loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
        frameon=True, framealpha=0.95, title='Legend', ncol=1
    )

    # Helpful zero line
    ax.axhline(0, color='#BBBBBB', linewidth=1)

    # Margins & layout
    ax.margins(x=0.01)
    plt.tight_layout()
    filename = f"grid_source_diffs_start{start_ts}_flex{str(fm).replace('-', 'neg')}.png"
    plt.savefig(IMAGE_DIR / filename, bbox_inches='tight', dpi=140)
    print(f"  -> Saved grid-source differences chart to {filename}")
    #plt.show()
    plt.close()

def save_heatmap_from_results(results_rows, csv_path: pathlib.Path, png_path: pathlib.Path):
    """
    Save all (timestep, flex_mag) results to CSV and create a heatmap image.
    X-axis: timeslot; Y-axis: flex magnitude; cell: Max_Duration_Min.
    """
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(csv_path, index=False)
    results_df['Timestep_Hours'] = (results_df['Timestep'] * 15) / 60
    results_df['Max_Duration_Hours'] = results_df['Max_Duration_Min'] / 60
    # Pivot for heatmap (rows: flex, cols: timestep)
    heat = results_df.pivot(index="Flex_Magnitude_kW", columns="Timestep_Hours", values="Max_Duration_Hours")

    # Make sure columns are sorted numerically for a left-to-right timeline
    heat = heat.reindex(sorted(heat.columns), axis=1)

    # Sort the index so negative flex magnitudes are at the bottom
    heat = heat.reindex(sorted(heat.index), axis=0)

    # Plot heatmap
    plt.figure(figsize=(20, 10))
    # Use a perceptually uniform colormap; NaN will appear as white
    ax = sns.heatmap(
        heat,
        cmap="viridis",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Max Duration (hours)"},
        square=False,
        annot=True,           # Show numbers in each square
        fmt=".0f",            # No decimals
        annot_kws={"size": 8} # Smaller font for clarity
    )
    ax.invert_yaxis()
    ax.set_xlabel("Timeslot (Hours)")
    ax.set_ylabel("Flex Magnitude (kW)")
    ax.set_title("Max Achievable Duration by Time (Hours) and Flex Magnitude")
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    #plt.show()
    plt.close()
    print(f"  -> Saved CSV to {csv_path.name}")
    print(f"  -> Saved heatmap to {png_path.name}")
