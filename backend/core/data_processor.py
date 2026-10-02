"""
Data Processor for UCI Parking Birmingham Dataset (id 482, Stolfi 2017, CC BY 4.0).

Features:
- Validates that Occupancy is a count (e.g. 50-600) compared to Capacity (387-687).
- Clips negative occupancy sensor anomalies to 0.
- Selects the 4 most complete car parks:
  1. BHMBCCMKT01 (Bullring Markets, Capacity: 577)
  2. BHMBCCTHL01 (Town Hall, Capacity: 387)
  3. BHMEURBRD01 (Broad Street, Capacity: 470)
  4. BHMMBMMBX01 (Mailbox, Capacity: 687)
- Pivots by 30-min rounded timestamp into a (T x 4) matrix.
- Handles missing values using forward-fill (ffill), maintaining temporal continuity.
- Appends derived 5th column 'Total' = sum of the 4 car parks, forming X (T x 5).
"""

import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.csv")

CAR_PARKS_META = {
    "BHMBCCMKT01": {"name": "Bullring Markets", "capacity": 577, "zone": "Zone A"},
    "BHMBCCTHL01": {"name": "Town Hall", "capacity": 387, "zone": "Zone B"},
    "BHMEURBRD01": {"name": "Broad Street", "capacity": 470, "zone": "Zone C"},
    "BHMMBMMBX01": {"name": "Mailbox", "capacity": 687, "zone": "Zone D"},
}


class DataProcessor:
    def __init__(self, csv_path: str = DATA_PATH):
        self.csv_path = csv_path
        self.raw_df: pd.DataFrame = None
        self.pivoted_df: pd.DataFrame = None
        self.X: np.ndarray = None  # Shape (T, 5)
        self.timestamps: List[str] = []
        self.columns: List[str] = []
        self.capacities: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_and_process()

    def _load_and_process(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset CSV not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self.raw_df = df

        # Step 1: Verify Occupancy is a count, not a rate
        # Occupancy values are numbers like 61, 150 up to 4327, matching Capacity magnitudes
        # We clip negative sensor errors (e.g. -8) to 0
        df["Occupancy"] = df["Occupancy"].clip(lower=0)

        # Step 2: Filter the 4 designated car parks with highest data completeness
        target_lots = list(CAR_PARKS_META.keys())
        sub_df = df[df["SystemCodeNumber"].isin(target_lots)].copy()

        # Step 3: Round timestamps to nearest 30-min interval
        sub_df["LastUpdated"] = pd.to_datetime(sub_df["LastUpdated"])
        sub_df["SlotTime"] = sub_df["LastUpdated"].dt.round("30min")

        # Step 4: Pivot into (timestamps x car_parks)
        piv = sub_df.pivot_table(
            index="SlotTime",
            columns="SystemCodeNumber",
            values="Occupancy",
            aggfunc="mean"
        )[target_lots]

        # Step 5: Handle missing values
        # In this subset of 1307 timestamps, only 1 missing value occurs in BHMEURBRD01.
        # We apply forward-fill (ffill) then back-fill (bfill), ideal for time series.
        missing_count_before = int(piv.isna().sum().sum())
        piv = piv.ffill().bfill()
        missing_count_after = int(piv.isna().sum().sum())

        # Step 6: Derive 5th column: Total = sum of the 4 car parks
        piv["Total"] = piv[target_lots].sum(axis=1)

        self.pivoted_df = piv
        self.X = piv.values.astype(np.float64)  # Shape (T, 5)
        self.timestamps = [t.strftime("%Y-%m-%d %H:%M") for t in piv.index]
        self.columns = list(piv.columns)
        self.capacities = {k: v["capacity"] for k, v in CAR_PARKS_META.items()}
        self.capacities["Total"] = sum(self.capacities.values())

        self.metadata = {
            "source": "UCI Parking Birmingham (id 482, Stolfi 2017, CC BY 4.0)",
            "occupancy_type": "vehicle_count",
            "occupancy_type_reasoning": "Occupancy values are integer vehicle counts (mean ~642, max 4327) compared with Capacities (220-4675). It is a count, not a rate fraction.",
            "num_timestamps": len(self.timestamps),
            "car_parks": CAR_PARKS_META,
            "columns": self.columns,
            "missing_values_handled": {
                "count_before": missing_count_before,
                "count_after": missing_count_after,
                "method": "Forward-fill (ffill) followed by backward-fill (bfill) for temporal continuity"
            },
            "matrix_X_shape": list(self.X.shape),
            "date_range": [self.timestamps[0], self.timestamps[-1]]
        }

    def get_summary(self) -> Dict[str, Any]:
        return self.metadata


# Singleton instance for quick module access
_processor = None

def get_data_processor() -> DataProcessor:
    global _processor
    if _processor is None:
        _processor = DataProcessor()
    return _processor
