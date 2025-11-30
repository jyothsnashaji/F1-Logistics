# constants.py

import pandas as pd

# Load unique circuits from circuits_mini.csv
CIRCUITS_CSV_PATH = "data/circuits.csv"
TRANSPORT_LINKS_CSV_PATH = "data/transport_links.csv"

# Load circuits
circuits_df = pd.read_csv(CIRCUITS_CSV_PATH)
AVAILABLE_CIRCUITS = circuits_df["circuit_id"].unique().tolist() if "circuit_id" in circuits_df.columns else circuits_df.iloc[:,0].unique().tolist()

# Load transport links
transport_links_df = pd.read_csv(TRANSPORT_LINKS_CSV_PATH)
TRANSPORT_LINKS = transport_links_df.to_dict(orient="records")

# Season start and end (example: pick first and last available circuit)
SEASON_START = AVAILABLE_CIRCUITS[0] if AVAILABLE_CIRCUITS else None
SEASON_END = AVAILABLE_CIRCUITS[-1] if AVAILABLE_CIRCUITS else None

MINIMUM_REST_DAYS = 2
TIME_TAKEN_PER_LINK = 1  # days
LEAD_TIME_MIN = 5  # days

TOTAL_NUM_REQUIRED_RACES = 12
MAX_DAYS_IN_SEASON = 300
NUM_DAYS_PER_RACE = 2
CRAZY_BUG_NUMBER = 100000000000