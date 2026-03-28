import scipy.io
import numpy as np
import csv
from pathlib import Path

# Configuration based on the paper
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'  # MATLAB files are expected in the data folder
TARGET_POINTS = 1300          # Required for the 13 means of 100 points


def to_scalar_string(value):
    """Unwrap MATLAB nested arrays into a lowercase Python string."""
    x = value
    while isinstance(x, np.ndarray):
        if x.size == 0:
            return ''
        x = x.flat[0]
    return str(x).strip().lower()


def resample_to_fixed_length(values, target_points):
    """
    Linear interpolation: converts ~300 points into 1300 points.
    This normalizes sequences before feeding them to the LSTM.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    # Create the original X axis (for example: 0 to 299)
    x_old = np.linspace(0.0, 1.0, num=values.size)
    # Create the target X axis (0 to 1299)
    x_new = np.linspace(0.0, 1.0, num=target_points)
    # Linear interpolation to fill gaps between measured points
    return np.interp(x_new, x_old, values)


def create_nasa_wide_csv(files, output_path):
    all_rows = []
    battery_summary = {}

    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"  WARNING: File not found, skipping: {file_path.name}")
            continue

        print(f"  Loading {file_path.name}...")
        mat = scipy.io.loadmat(file_path)
        # Dynamically extract battery name (B0005, B0006, etc.)
        b_name = [k for k in mat.keys() if not k.startswith('__')][0]
        cycles = mat[b_name][0, 0]['cycle'][0]

        discharge_count = 0
        skipped_count = 0

        for i, cycle in enumerate(cycles):
            # Strictly keep discharge cycles only
            if to_scalar_string(cycle['type']) != 'discharge':
                continue

            try:
                data = cycle['data'][0, 0]

                # Measured voltage — primary feature
                voltages = data['Voltage_measured'][0]
                v_seq = resample_to_fixed_length(voltages, TARGET_POINTS)
                if v_seq is None:
                    skipped_count += 1
                    continue

                # Real capacity — SOH target label (Y)
                capacity = data['Capacity'][0][0]

                # Use sequential discharge index (1-based), not raw MATLAB cycle index
                row = [b_name, discharge_count + 1, round(float(capacity), 6)] + v_seq.tolist()
                all_rows.append(row)
                discharge_count += 1

            except (KeyError, IndexError) as e:
                print(f"    WARNING: Skipping cycle {i} in {b_name}: {e}")
                skipped_count += 1

        battery_summary[b_name] = {
            'discharge_cycles': discharge_count,
            'skipped': skipped_count
        }
        print(f"    OK: {b_name}: {discharge_count} discharge cycles imported, {skipped_count} skipped.")

    # Create headers (Battery, Cycle, Capacity, V1, V2... V1300)
    columns = ['Battery', 'Cycle', 'Capacity'] + [f'V{j+1}' for j in range(TARGET_POINTS)]

    output_path = Path(output_path)
    print(f"\n  Writing CSV to {output_path.name}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(all_rows)

    # Final import report
    print(f"\n{'='*55}")
    print(f"  IMPORT SUMMARY")
    print(f"{'='*55}")
    for b, s in battery_summary.items():
        status = "OK" if s['skipped'] == 0 else f"{s['skipped']} SKIPPED"
        print(f"  {b:6s}  {s['discharge_cycles']:4d} discharge cycles   [{status}]")
    print(f"{'-'*55}")
    print(f"  TOTAL   {len(all_rows):4d} rows")
    print(f"  COLUMNS {len(columns):4d}  (Battery, Cycle, Capacity + {TARGET_POINTS} voltage pts)")
    print(f"  OUTPUT  {output_path}")
    print(f"{'='*55}")


if __name__ == '__main__':
    mat_files = [
        DATA_DIR / 'B0005.mat',
        DATA_DIR / 'B0006.mat',
        DATA_DIR / 'B0007.mat',
        DATA_DIR / 'B0018.mat',
    ]
    create_nasa_wide_csv(mat_files, DATA_DIR / 'dataset_soh_nasa.csv')
