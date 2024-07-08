import argparse
import os
import pandas as pd
import numpy as np
import h5py

"""
This script is designed to ensure the accuracy and consistency of event timestamps within Electronic Health Record (EHR) datasets, particularly focusing on the MIMIC-IV dataset. The script performs comprehensive verification through two main tasks:

1. **Timestamp Verification between Cohort and ICU Stays Data:**
   - It compares the `intime` and `outtime` values in the `{ehr}_cohorts.csv` file with those in the `icustays.csv` or `patients.csv` files. This ensures that these crucial timestamps are identical across different files, verifying there is no discrepancy or data drift. The alignment of these timestamps is critical for maintaining the integrity of the data used in medical research.

2. **Event Time Alignment within HDF5 File:**
   - The script verifies that the events recorded in the HDF5 file (`.h5`) are within the expected observation window. For instance, in a 24-hour observation window, if the `intime` is 1/1 00:00, it checks that events fall within 1/1 00:00 to 1/2 00:00, rather than being shifted to 1/1 09:00 to 1/2 09:00. This step is crucial to ensure the temporal accuracy of events, as any shift could lead to incorrect analysis.

To accomplish these tasks, the script:
- Loads the required CSV files (`inputevents.csv`, `icustays.csv`, and `{ehr}_cohorts.csv`).
- Samples around 50 patient IDs to manage computational load while ensuring a representative sample.
- Calculates the ICU stay duration in minutes and ensures that `intime` and `outtime` are consistent.
- Filters `inputevents` to focus on the relevant observation window from the `intime`.
- Identifies the earliest event start times for each patient and validates their consistency with the dataset.
- Reads the EHR data from the HDF5 file and confirms that the event timestamps align with the `intime` without any drift.

Finally, the script outputs the verification results, confirming the correctness of initial events and their counts. This rigorous validation process ensures the reliability of the dataset, which is essential for accurate and trustworthy medical research and analysis.
"""

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ehr",
        type=str,
        required=True,
        choices=["mimiciii", "mimiciv", "eicu"],
        help="Name of the EHR dataset",
    )
    parser.add_argument(
        "--obs_size", type=int, default=12, help="Observation window size by the hour"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the original data source"
    )
    parser.add_argument(
        "--dest", type=str, required=True, help="Path to the preprocessed data"
    )

    return parser

def main():
    args = get_parser().parse_args()
    pid = "stay_id" if args.ehr == "mimiciv" else "patientunitstayid"

    # Load datasets
    inputevents = pd.read_csv(os.path.join(args.data, "icu/inputevents.csv"))
    icustays = pd.read_csv(os.path.join(args.data, "icu/icustays.csv"))
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])

    data = h5py.File(os.path.join(args.dest, f"{args.ehr}.h5"), "r")['ehr']
    cohorts = pd.read_csv(os.path.join(args.dest, f"{args.ehr}_cohort.csv"))
    
    # Sample 50 patient IDs
    sample_ids = cohorts[pid].sample(n=50, random_state=1).values

    # Process cohorts
    cohort_sampled = cohorts[cohorts[pid].isin(sample_ids)][[pid, "INTIME", "OUTTIME"]]
    cohort_sampled['INTIME'] = pd.to_datetime(cohort_sampled['INTIME'].astype(str).str.replace("+00:00", "", regex=False))
    cohort_sampled['OUTTIME'] = cohort_sampled['OUTTIME'].astype(int)

    # Calculate ICU stay duration in minutes
    icustays_sampled = icustays[icustays[pid].isin(sample_ids)][[pid, "intime", "outtime"]]
    icustays_sampled['outtime'] = (icustays_sampled['outtime'] - icustays_sampled['intime']).dt.total_seconds() / 60
    icustays_sampled['outtime'] = icustays_sampled['outtime'].astype(int)

    # Merge cohort and ICU stay data
    merged_df = pd.merge(cohort_sampled, icustays_sampled, on=pid, how='inner', suffixes=('_cohort', '_icu'))
    merged_df['intime_match'] = merged_df['INTIME'] == merged_df['intime']
    merged_df['outtime_match'] = merged_df['OUTTIME'] == merged_df['outtime']

    # Ensure the matches are correct
    assert merged_df.intime_match.sum() == merged_df.outtime_match.sum() == len(merged_df)

    # Process input events
    inputevents_sampled = inputevents[inputevents[pid].isin(sample_ids)][[pid, "starttime"]]
    inputevents_sampled['starttime'] = pd.to_datetime(inputevents_sampled['starttime'])

    # Merge and filter input events based on observation window
    merged_df = pd.merge(inputevents_sampled, cohort_sampled, on=pid)
    filtered_df = merged_df[(merged_df['starttime'] >= merged_df['INTIME']) &
                            (merged_df['starttime'] <= merged_df['INTIME'] + pd.Timedelta(hours=args.obs_size))]
    filtered_df = filtered_df[[pid, 'starttime']]

    # Find the earliest event start time for each patient
    min_starttimes = filtered_df.groupby(pid)['starttime'].min().reset_index()

    # Count the number of earliest events for each patient
    earliest_events = pd.merge(filtered_df, min_starttimes, on=[pid, 'starttime'], how='inner')
    earliest_event_counts = earliest_events.groupby(pid).size().reset_index(name='earliest_event_count')

    # Validate the data
    for _sample_id in sample_ids:
        time_data = data[str(_sample_id)]["time"][:]
        hi_data = data[str(_sample_id)]["hi"][:, 0, 0][:len(time_data)]

        # Find the index where hi_data equals 7758
        inputevent_start_idx = np.where(hi_data == 7758)[0]

        if inputevent_start_idx.size > 0:
            start_time = time_data[inputevent_start_idx[0]]

            # Count occurrences where time matches start_time and hi_data equals 7758
            count = np.logical_and(time_data == start_time, hi_data == 7758).sum()

            print(f"ID: {_sample_id}, Start Time: {start_time}, Count: {count}")
            assert count == earliest_event_counts[earliest_event_counts[pid] == _sample_id]['earliest_event_count'].values[0]

if __name__ == "__main__":
    main()
