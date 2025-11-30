import pandas as pd
from constants import *

class DataLoader:
    def __init__(self):
        # 1. Load Circuits (Directly from Constants)
        self.circuits = AVAILABLE_CIRCUITS
        
        # 2. Build Revenue Map & ID-to-Name Map
        df_circuits = circuits_df.copy()
        df_circuits.columns = [c.strip() for c in df_circuits.columns]
        
        # Determine ID column and Name column
        id_col = "circuit_id" if "circuit_id" in df_circuits.columns else df_circuits.columns[0]
        # Try to find a 'Circuit' or similar column for names
        name_col = next((c for c in df_circuits.columns if 'Circuit' in c), id_col)
        
        self.id_to_name = df_circuits.set_index(id_col)[name_col].to_dict()
        
        rev_col = next((c for c in df_circuits.columns if 'Hosting' in c), None)
        if rev_col:
            if df_circuits[rev_col].dtype == 'O':
                df_circuits[rev_col] = df_circuits[rev_col].replace('[\$,]', '', regex=True).astype(float)
            self.hosting_fees = df_circuits.set_index(id_col)[rev_col].to_dict()
        else:
            self.hosting_fees = {c: 0 for c in self.circuits}

        # 3. Load Transport Links
        self.transport_links = TRANSPORT_LINKS
        
        # 4. Build Dense Maps
        self.cost_map = {}
        self.emission_map = {}
        self.time_map = {}
        self.keys = [] 
        
        modes = ['Air', 'Road', 'Sea']
        
        def get_val(row, potential_keys):
            for k in potential_keys:
                if k in row: return row[k]
            return None

        sparse_lookup = {}
        for row in self.transport_links:
            u = get_val(row, ['from_id', 'origin_circuit_i', 'from'])
            v = get_val(row, ['to_id', 'dest_circuit_j', 'to'])
            m = get_val(row, ['mode', 'mode_m'])
            
            if u and v and m:
                c = get_val(row, ['cost', 'cost_Cijm_USD'])
                e = get_val(row, ['emission', 'emissions_Eijm_kgCO2e'])
                
                sparse_lookup[(u, v, m)] = {
                    'cost': float(c) if c is not None else CRAZY_BUG_NUMBER,
                    'emission': float(e) if e is not None else CRAZY_BUG_NUMBER
                }

        for i in self.circuits:
            for j in self.circuits:
                if i == j: continue
                for m in modes:
                    key = (i, j, m)
                    self.keys.append(key)
                    if key in sparse_lookup:
                        self.cost_map[key] = sparse_lookup[key]['cost']
                        self.emission_map[key] = sparse_lookup[key]['emission']
                    else:
                        self.cost_map[key] = CRAZY_BUG_NUMBER
                        self.emission_map[key] = CRAZY_BUG_NUMBER
                    self.time_map[key] = float(TIME_TAKEN_PER_LINK)

    # Getters
    def load_circuits(self): return self.circuits
    def get_circuit_names(self): return self.id_to_name
    def get_link_costs(self): return self.cost_map
    def get_link_emissions(self): return self.emission_map
    def get_link_times(self): return self.time_map
    def get_circuit_revenue(self): return self.hosting_fees
    
    def get_season_start_end(self): return SEASON_START, SEASON_END
    def get_minimum_rest_days(self): return MINIMUM_REST_DAYS
    def get_lead_time_min(self): return LEAD_TIME_MIN
    def get_total_num_required_races(self): return TOTAL_NUM_REQUIRED_RACES
    def get_max_season_days(self): return MAX_DAYS_IN_SEASON
    def get_num_days_per_race(self): return NUM_DAYS_PER_RACE