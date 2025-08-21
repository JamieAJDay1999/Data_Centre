import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import pathlib
import seaborn as sns
# MODIFIED: Switched from pulp to pyomo
import pyomo.environ as pyo

IMAGE_DIR = pathlib.Path("static/images/flexibility_outputs")

# Create directories if they don't exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def extract_detailed_results(m, params, data, start_timestep, flex_time, baseline_df):
    """
    MODIFIED: This function is now compatible with a solved Pyomo model object
    using explicit unit-based variable names.
    It extracts power, energy, temperature, and CPU data.
    """
    time_slots = sorted(list(m.TEXT_SLOTS.ordered_data()))
    results_df = pd.DataFrame(index=time_slots)
    results_df.index.name = 'Time_Slot_EXT'

    final_columns = []

    # --- Power Metrics ---
    # Calculate total power in kW, converting chiller power from W to kW
    results_df['P_Total_kw_Opt'] = [
        pyo.value(m.p_grid_it_kw[s]) +
        (pyo.value(m.p_chiller_hvac_w[s]) / 1000.0) +
        (pyo.value(m.p_chiller_tes_w[s]) / 1000.0) +
        pyo.value(m.p_grid_od_kw[s]) +
        pyo.value(m.p_ups_ch_kw[s])
        for s in time_slots
    ]
    results_df['P_Total_kw_Base'] = [baseline_df.loc[s, 'P_Total_kW'] if s in baseline_df.index else 0 for s in time_slots]
    results_df['P_Total_kw_Diff'] = results_df['P_Total_kw_Opt'] - results_df['P_Total_kw_Base']
    final_columns.extend(['P_Total_kw_Base', 'P_Total_kw_Opt', 'P_Total_kw_Diff'])
    
    power_source_map = {
        'P_IT_Total_kW': m.p_it_total_kw, 'P_Grid_IT_kW': m.p_grid_it_kw,
        'P_Chiller_HVAC_kW': m.p_chiller_hvac_w, 'P_Chiller_TES_kW': m.p_chiller_tes_w,
        'P_Grid_Other_kW': m.p_grid_od_kw, 'P_UPS_Charge_kW': m.p_ups_ch_kw,
        'P_UPS_Discharge_kW': m.p_ups_disch_kw,
    }

    for source_name, var_obj in power_source_map.items():
        # Chiller power is in Watts in the model, so convert to kW for reporting
        if 'Chiller' in source_name:
            opt_vals = [pyo.value(var_obj[s]) / 1000.0 for s in time_slots]
        else: # Other power sources are already in kW
            opt_vals = [pyo.value(var_obj[s]) for s in time_slots]

        base_vals = [baseline_df.loc[s, source_name] if s in baseline_df.index else 0 for s in time_slots]
        diff_vals = [opt - base for opt, base in zip(opt_vals, base_vals)]
        base_col, opt_col, diff_col = f'{source_name}_base', f'{source_name}_opt', f'{source_name}_diff'
        results_df[base_col] = base_vals
        results_df[opt_col] = opt_vals
        results_df[diff_col] = diff_vals
        final_columns.extend([base_col, opt_col, diff_col])

    hvac_opt_kw = [pyo.value(m.p_chiller_hvac_w[s]) / 1000.0 for s in time_slots]
    tes_opt_kw = [pyo.value(m.p_chiller_tes_w[s]) / 1000.0 for s in time_slots]
    opt_vals_cool_kw = [h + t for h, t in zip(hvac_opt_kw, tes_opt_kw)]
    base_vals_cool_kw = [baseline_df.loc[s, 'P_Grid_Cooling_kW'] if s in baseline_df.index else 0 for s in time_slots]
    diff_vals_cool_kw = [opt - base for opt, base in zip(opt_vals_cool_kw, base_vals_cool_kw)]
    results_df['P_Grid_Cooling_kW_base'] = base_vals_cool_kw
    results_df['P_Grid_Cooling_kW_opt'] = opt_vals_cool_kw
    results_df['P_Grid_Cooling_kW_diff'] = diff_vals_cool_kw
    final_columns.extend(['P_Grid_Cooling_kW_base', 'P_Grid_Cooling_kW_opt', 'P_Grid_Cooling_kW_diff'])

    # --- Energy Storage Metrics ---
    e_tes_opt_kwh = [pyo.value(m.e_tes_kwh[s]) for s in time_slots]
    e_tes_base_kwh = [baseline_df.loc[s, 'E_TES_kWh'] if s in baseline_df.index and 'E_TES_kWh' in baseline_df.columns else 0 for s in time_slots]
    results_df['E_TES_kWh_base'] = e_tes_base_kwh
    results_df['E_TES_kWh_opt'] = e_tes_opt_kwh
    results_df['E_TES_kWh_diff'] = [opt - base for opt, base in zip(e_tes_opt_kwh, e_tes_base_kwh)]
    final_columns.extend(['E_TES_kWh_base', 'E_TES_kWh_opt', 'E_TES_kWh_diff'])

    # --- Temperature Metrics ---
    temp_source_map = {
        'T_IT_Celsius': m.t_it, 'T_Rack_Celsius': m.t_rack,
        'T_ColdAisle_Celsius': m.t_cold_aisle, 'T_HotAisle_Celsius': m.t_hot_aisle
    }
    
    for base_name, var_obj in temp_source_map.items():
        opt_vals = [pyo.value(var_obj[s]) for s in time_slots]
        base_vals = [baseline_df.loc[s, base_name] if s in baseline_df.index and base_name in baseline_df.columns else 0 for s in time_slots]
        diff_vals = [opt - base for opt, base in zip(opt_vals, base_vals)]
        base_col, opt_col, diff_col = f'{base_name}_base', f'{base_name}_opt', f'{base_name}_diff'
        results_df[base_col] = base_vals
        results_df[opt_col] = opt_vals
        results_df[diff_col] = diff_vals
        final_columns.extend([base_col, opt_col, diff_col])

    # --- Cost and Nominal Load Metrics ---
    results_df['Price_GBP_per_MWh'] = [data['electricity_price'][s] if s < len(data['electricity_price']) else 0 for s in time_slots]
    results_df['P_IT_Nominal'] = [data['Pt_IT_nom_TEXT'][s] if s < len(data['Pt_IT_nom_TEXT']) else 0 for s in time_slots]
    final_columns.extend(['Price_GBP_per_MWh', 'P_IT_Nominal'])

    # --- CPU Load Metrics ---
    results_df['Inflexible_Load_CPU_Nom'] = [data['inflexibleLoadProfile_TEXT'][s] if s < len(data['inflexibleLoadProfile_TEXT']) else 0 for s in time_slots]
    results_df['Flexible_Load_CPU_Nom'] = [data['flexibleLoadProfile_TEXT'][s] if s < len(data['flexibleLoadProfile_TEXT']) else 0 for s in time_slots]

    total_cpu_load_opt = [pyo.value(m.total_cpu[s]) for s in time_slots]
    total_cpu_load_base = [baseline_df.loc[s, 'Total_CPU_Load'] if s in baseline_df.index else 0 for s in time_slots]
    
    flexible_cpu_usage = {s: 0 for s in time_slots}
    for (t_job, k_job, s_job), var in m.ut_ks.items():
        # CORRECTED LINE: Removed the unsupported keyword argument
        val = pyo.value(var)
        if val is not None and val > 1e-6:
            if s_job in flexible_cpu_usage:
                flexible_cpu_usage[s_job] += val

    flexible_cpu_usage_list = [flexible_cpu_usage.get(s, 0) for s in time_slots]
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

def plot_flex_contribution_grid(all_plot_data, timesteps, flex_magnitudes):
    """
    Produces a figure with n*m stacked bar charts, where n is the number of
    different magnitudes and m is the number of different timesteps.
    """
    sorted_flex_mags = sorted(flex_magnitudes, reverse=True)
    n_rows, n_cols = len(sorted_flex_mags), len(timesteps)
    if n_rows == 0 or n_cols == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharey=True, squeeze=False)

    col_map = {
        'P_Grid_IT_kW_diff': 'IT (grid)', 'P_Chiller_HVAC_kW_diff': 'Chiller HVAC',
        'P_Chiller_TES_kW_diff': 'Chiller TES', 'P_Grid_Other_kW_diff': 'Other overhead',
        'P_UPS_Charge_kW_diff': 'UPS charge',
    }
    palette = {
        'IT (grid)': '#0072B2', 'Chiller HVAC': '#E69F00', 'Chiller TES': '#009E73',
        'Other overhead': '#CC79A7', 'UPS charge': '#56B4E9',
    }
    series_order = list(col_map.values())

    ts_map = {ts: i for i, ts in enumerate(timesteps)}
    fm_map = {fm: i for i, fm in enumerate(sorted_flex_mags)}
    global_max_dur = max(d.get('dur_steps', 0) for d in all_plot_data) if all_plot_data else 0

    for plot_data in all_plot_data:
        results_df, start_ts, fm, dur_steps = plot_data['results_df'], plot_data['ts'], plot_data['fm'], plot_data['dur_steps']
        if start_ts not in ts_map or fm not in fm_map: continue
        row_idx, col_idx = fm_map[fm], ts_map[start_ts]
        ax = axes[row_idx, col_idx]

        t0, t1 = start_ts, start_ts + dur_steps
        window_idx = [t for t in results_df.index if t0 <= t < t1]
        if not window_idx:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        cols_present = [c for c in col_map if c in results_df.columns]
        diff_df = results_df.loc[window_idx, cols_present].rename(columns=col_map)
        
        x = np.arange(len(diff_df))
        pos_bottom = np.zeros(len(diff_df))
        neg_bottom = np.zeros(len(diff_df))

        for name in series_order:
            if name not in diff_df.columns: continue
            s = diff_df[name].values
            color = palette.get(name)
            s_pos, s_neg = np.clip(s, 0, None), np.clip(s, None, 0)
            ax.bar(x, s_pos, width=0.82, bottom=pos_bottom, color=color, edgecolor='white', linewidth=0.6)
            pos_bottom += s_pos
            ax.bar(x, s_neg, width=0.82, bottom=neg_bottom, color=color, edgecolor='white', linewidth=0.6)
            neg_bottom += s_neg

        sum_diffs = diff_df.sum(axis=1).values
        ax.scatter(x, sum_diffs, marker='x', s=64, color='#666666', linewidths=1.5)
        ax.axhline(fm, color='black', alpha=0.5, linewidth=2, linestyle='--')
        ax.axhline(0, color='#BBBBBB', linewidth=1)
        ax.grid(axis='y', linestyle=':', color='#E0E0E0')
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=0.01)

        if global_max_dur > 0:
            longest_x = np.arange(global_max_dur)
            duration_labels_hours = np.arange(1, global_max_dur + 1) / 4.0
            tick_every = max(1, len(longest_x) // 8)
            ax.set_xlim(-0.5, global_max_dur - 0.5)
            if tick_every > 0 and len(longest_x) > 0:
                ax.set_xticks(longest_x[::tick_every])
                ax.set_xticklabels([f"{t:.2f}" for t in duration_labels_hours[::tick_every]], rotation=45, ha='right', fontsize=10)
            elif len(longest_x) > 0:
                ax.set_xticks(longest_x)
                ax.set_xticklabels([f"{t:.2f}" for t in duration_labels_hours], rotation=45, ha='right', fontsize=10)

    for r in range(n_rows):
        axes[r, 0].set_ylabel(f'{sorted_flex_mags[r]} kW', rotation=0, va="center", ha="right", labelpad=5, fontsize=13)
    for c_idx, ts in enumerate(timesteps):
        hour, minute = int(ts // 4), int((ts % 4) * 15)
        axes[0, c_idx].set_title(f'Start Time: {hour:02d}:{minute:02d}', fontsize=12, pad=2)

    legend_swatches = [Patch(facecolor=c, edgecolor='w', label=n) for n, c in palette.items()]
    extra_items = [Line2D([0], [0], ls='--', c='k', alpha=0.5, lw=2, label='Flex target'),
                   Line2D([0], [0], marker='x', ls='None', c='#666', ms=8, label='Sum of diffs')]
    fig.legend(handles=legend_swatches + extra_items, loc='upper left', bbox_to_anchor=(0.87, 0.95),
               borderaxespad=0, title='Legend', fontsize=10, title_fontsize=14)

    fig.text(0.5, 0.10, 'Flexibility Duration (Hours)', ha='center', va='center', fontsize=16, weight='bold')
    fig.text(0.01, 0.5, 'Magnitude (kW)', ha='center', va='center', rotation='vertical', fontsize=16, weight='bold')
    fig.suptitle('Grid Power Change by Source', fontsize=20, weight='bold')
    fig.subplots_adjust(left=0.12, right=0.85, top=0.9, bottom=0.15, wspace=0.08, hspace=0.12)

    filename = "grid_source_diffs_summary.png"
    plt.savefig(IMAGE_DIR / filename, dpi=140)
    plt.show()
    print(f"  -> Saved grid summary chart to {filename}")
    plt.close()

def save_heatmap_from_results(results_rows, csv_path: pathlib.Path, png_path: pathlib.Path):
    """
    Save all (timestep, flex_mag) results to CSV and create a heatmap image.
    """
    results_df = pd.DataFrame(results_rows)
    results_df['Timestep_Hours'] = (results_df['Timestep'] * 15) / 60
    results_df['Max_Duration_Hours'] = results_df['Max_Duration_Min'] / 60
    results_df.to_csv(csv_path, index=False)
    heat = results_df.pivot(index="Flex_Magnitude_kW", columns="Timestep_Hours", values="Max_Duration_Hours")
    heat = heat.reindex(sorted(heat.columns), axis=1)
    heat = heat.reindex(sorted(heat.index), axis=0)

    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(heat, cmap="viridis", linewidths=0.3, linecolor="white",
                     cbar_kws={"label": "Max Duration (hours)"}, square=False,
                     annot=True, fmt=".1f", annot_kws={"size": 8})
    ax.invert_yaxis()
    ax.set_xlabel("Timeslot (Hours into Day)")
    ax.set_ylabel("Flex Magnitude (kW)")
    ax.set_title("Max Achievable Duration by Time and Flex Magnitude (Hours)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved CSV to {csv_path.name}")
    print(f"  -> Saved heatmap to {png_path.name}")