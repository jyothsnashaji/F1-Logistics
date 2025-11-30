# data_loader.py
import pandas as pd
from constants import *

class DataLoader:
    def __init__(self):
        # Load available circuits from CSV
        circuits_df = pd.read_csv(CIRCUITS_CSV_PATH)
        if "circuit_id" in circuits_df.columns:
            self.circuits = circuits_df["circuit_id"].unique().tolist()
        else:
            self.circuits = circuits_df.iloc[:, 0].unique().tolist()

        # Load transport links from CSV
        transport_links_df = pd.read_csv(TRANSPORT_LINKS_CSV_PATH)
        self.transport_links = transport_links_df.to_dict(orient="records")

    def load_circuits(self):
        """Returns the list of available circuits."""
        return self.circuits

    def load_transport_links(self):
        """Returns the transport links as a list of dicts."""
        return self.transport_links

    def get_season_start_end(self):
        """Returns the season start and end circuit IDs."""
        return SEASON_START, SEASON_END

    def get_minimum_rest_days(self):
        return MINIMUM_REST_DAYS

    def get_lead_time_min(self):
        return LEAD_TIME_MIN

    def get_time_taken_per_link(self):
        return TIME_TAKEN_PER_LINK
