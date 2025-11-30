# data_loader.py
import pandas as pd
import math
from constants import *

class DataLoader:
    def __init__(self):
        # Load available circuits from CSV
        circuits_df = pd.read_csv(CIRCUITS_CSV_PATH)
        self.circuits_df = circuits_df.copy()
        print(circuits_df.columns)
        if "circuit_id" in circuits_df.columns:
            self.circuits = circuits_df["circuit_id"].unique().tolist()
        else:
            self.circuits = circuits_df.iloc[:, 0].unique().tolist()
        self.hosting_fees = circuits_df.set_index("circuit_id")["Hosting Fees"].to_dict()
        # Load transport links from CSV
        transport_links_df = pd.read_csv(TRANSPORT_LINKS_CSV_PATH)
        self.transport_links = transport_links_df.to_dict(orient="records")
        
        # Build dense maps for all (from_id, to_id, mode) combinations,
        # with +inf for missing cost/emission, uniform time for all.
        modes = ['Air', 'Road', 'Sea']
        # Sparse values from CSV
        cost_sparse = {(l['from_id'], l['to_id'], l['mode']): float(l['cost']) for l in self.transport_links}
        emission_sparse = {(l['from_id'], l['to_id'], l['mode']): float(l['emission']) for l in self.transport_links}
        # Dense maps
        self.cost_map = {}
        self.emission_map = {}
        self.time_map = {}
        self.keys  = []
        for i in self.circuits:
            for j in self.circuits:
                if i == j:
                    continue
                for m in modes:
                    key = (i, j, m)
                    self.keys.append(key)
                    self.cost_map[key] = cost_sparse.get(key, CRAZY_BUG_NUMBER)
                    self.emission_map[key] = emission_sparse.get(key, CRAZY_BUG_NUMBER)
                    self.time_map[key] = float(TIME_TAKEN_PER_LINK)

    def load_circuits(self):
        """Returns the list of available circuits."""
        return self.circuits

    def load_transport_links(self):
        """Returns the transport links as a list of dicts."""
        return self.transport_links

    def get_link_costs(self):
        """Dense cost map over all (from_id,to_id,mode); missing combos are +inf."""
        return self.cost_map

    def get_link_emissions(self):
        """Dense emission map over all (from_id,to_id,mode); missing combos are +inf."""
        return self.emission_map

    def get_link_times(self):
        """Dense time map over all (from_id,to_id,mode); uniform per link."""
        return self.time_map

    def get_season_start_end(self):
        """Returns the season start and end circuit IDs."""
        return SEASON_START, SEASON_END

    def get_minimum_rest_days(self):
        return MINIMUM_REST_DAYS

    def get_lead_time_min(self):
        return LEAD_TIME_MIN

    def get_time_taken_per_link(self):
        return TIME_TAKEN_PER_LINK
    
    def get_total_num_required_races(self):
        return TOTAL_NUM_REQUIRED_RACES
    
    def get_max_season_days(self):
        """Returns the maximum number of days in the season."""
        return MAX_DAYS_IN_SEASON

    def get_num_days_per_race(self):
        """Returns the number of days allocated per race."""
        return NUM_DAYS_PER_RACE

    def get_circuit_revenue(self):
        """Returns a dictionary mapping circuit IDs to their revenue."""
        # For simplicity, let's assume revenue is proportional to circuit index
        return self.hosting_fees

if __name__ == "__main__":
    data_loader = DataLoader()
    print("Available Circuits:", data_loader.load_circuits())
    print("Transport Links:", data_loader.load_transport_links())
    print("Season Start and End:", data_loader.get_season_start_end())
    print("Minimum Rest Days:", data_loader.get_minimum_rest_days())
    print("Lead Time Minimum:", data_loader.get_lead_time_min())
    print("Time Taken Per Link:", data_loader.get_time_taken_per_link())
    print("Total Number of Required Races:", data_loader.get_total_num_required_races())
    print("Max Days in Season:", data_loader.get_max_season_days())
    print("Number of Days Per Race:", data_loader.get_num_days_per_race())
    print("Circuit Revenue:", data_loader.get_circuit_revenue())