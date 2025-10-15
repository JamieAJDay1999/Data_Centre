import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Read the CSV file efficiently, parsing timestamps
# Assuming the file is named 'inputs/ukpn-data-centre-demand-profiles.csv' - change if necessary
df = pd.read_csv('inputs/ukpn-data-centre-demand-profiles.csv', parse_dates=['local_timestamp', 'utc_timestamp'])

# Ensure the data is sorted by utc_timestamp within each data center for proper plotting
df = df.sort_values(['anonymised_data_centre_name', 'utc_timestamp'])

# Calculate variance of utilisation ratio for each data centre
variance_df = df.groupby('anonymised_data_centre_name')['hh_utilisation_ratio'].var().reset_index()
variance_df.columns = ['anonymised_data_centre_name', 'variance']

# Select top 5 highest variance (most deviation)
top_5_most = variance_df.nlargest(5, 'variance')['anonymised_data_centre_name'].tolist()

# Select top 5 lowest variance (least deviation)
top_5_least = variance_df.nsmallest(5, 'variance')['anonymised_data_centre_name'].tolist()

# Combine all selected centres
selected_centres = top_5_most + top_5_least

print(f"Top 5 most deviating: {top_5_most}")
print(f"Top 5 least deviating: {top_5_least}")

# First plot: Daily average utilisation over entire time window
daily_avg_df = df[df['anonymised_data_centre_name'].isin(selected_centres)].copy()
daily_avg_df['utc_timestamp'] = pd.to_datetime(daily_avg_df['utc_timestamp'], utc=True)
daily_avg_df['date'] = daily_avg_df['utc_timestamp'].dt.date
daily_avg = daily_avg_df.groupby(['anonymised_data_centre_name', 'date'])['hh_utilisation_ratio'].mean().reset_index()
daily_avg['date'] = pd.to_datetime(daily_avg['date'])

plt.figure(figsize=(15, 8))

for name in selected_centres:
    centre_data = daily_avg[daily_avg['anonymised_data_centre_name'] == name]
    var = variance_df[variance_df['anonymised_data_centre_name'] == name]['variance'].iloc[0]
    label = f"{name} (var: {var:.4f})"
    plt.plot(centre_data['date'], centre_data['hh_utilisation_ratio'], 
             label=label, linewidth=1, alpha=0.8)

plt.xlabel('Date')
plt.ylabel('Daily Average HH Utilisation Ratio')
plt.title('Daily Average Utilisation Ratio Over Time: 5 Most & Least Deviating Data Centres')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Second plot: Original hourly data for a hard-coded random 2 weeks in 2023 (e.g., May 15-28, 2023)
# with three subplots stacked vertically, each showing all 10 data centres
# Set a random seed for reproducibility
np.random.seed(42)

# Generate three unique random start dates in 2023
all_days_2023 = pd.date_range('2023-01-01', '2023-12-17', freq='D', tz='UTC')  # End at Dec 17 to allow 14 days
random_start_dates = np.random.choice(all_days_2023, size=3, replace=False)
random_start_dates = sorted(random_start_dates)

import matplotlib.dates as mdates

# Prepare three DataFrames, one for each random period
period_dfs = []
for start_date in random_start_dates:
    end_date = start_date + timedelta(days=14)
    period_df = df[
        (df['anonymised_data_centre_name'].isin(selected_centres)) &
        (df['utc_timestamp'] >= start_date) &
        (df['utc_timestamp'] < end_date)
    ].copy()
    period_dfs.append((start_date, end_date, period_df))

# Create figure with three subplots stacked vertically
fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=False, sharey=True)
axs = axs.flatten()

# Set a single title for the entire figure

for i, (ax, (start_date, end_date, period_df)) in enumerate(zip(axs, period_dfs)):
    for name in selected_centres:
        centre_data = period_df[period_df['anonymised_data_centre_name'] == name]
        var = variance_df[variance_df['anonymised_data_centre_name'] == name]['variance'].iloc[0]
        plot_label = f"{name} (var: {var:.4f})"
        ax.plot(centre_data['utc_timestamp'], centre_data['hh_utilisation_ratio'],
                label=plot_label, linewidth=0.8, alpha=0.6)
    # Format x-axis to show date as 'Mon DD' (no year), horizontal
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_ylabel('HH Utilisation Ratio')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=0)
    # Add a subtitle for each subplot indicating the period
    if i == 0:
        ax.set_title('Hourly Utilisation Ratio for 10 Data Centres over 2 Week Period', fontsize=16, y=1.02)

axs[-1].set_xlabel('UTC Timestamp')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()