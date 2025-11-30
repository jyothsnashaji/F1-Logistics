import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from collections import deque
from data_loader import DataLoader
from constants import *

# ==========================================
# 1. CORE OPTIMIZATION (AUGMECON / MILP)
# ==========================================

def create_and_solve_gurobi_milp(data_loader, objective_index, number_of_solutions, eps_values=None, R_ranges=None):
    """
    Solves the MILP and returns the Top N solutions found (Solution Pool).
    """
    found_solutions = []
    
    try:
        m = gp.Model("F1_Logistics")
        m.setParam('OutputFlag', 0)
        
        # --- ENABLE SOLUTION POOL ---
        # 2 = Search for n best solutions
        m.setParam(GRB.Param.PoolSearchMode, 2) 
        # Limit pool size to top 10 solutions per run
        m.setParam(GRB.Param.PoolSolutions, number_of_solutions) 
        # Gap tolerance (relative) to accept slightly suboptimal solutions
        m.setParam(GRB.Param.PoolGap, 0.20) # Accept solutions within 20% of optimal

        V = data_loader.load_circuits()
        S, E = data_loader.get_season_start_end()
        K = data_loader.get_total_num_required_races()
        T_max = data_loader.get_max_season_days()
        T_race = data_loader.get_num_days_per_race()
        T_rest_min = data_loader.get_minimum_rest_days()
        L_min = data_loader.get_lead_time_min()
        
        links = data_loader.keys
        R_i = data_loader.get_circuit_revenue()
        id_map = data_loader.get_circuit_names()

        coeffs = {
            'C': data_loader.get_link_costs(),
            'E': data_loader.get_link_emissions(),
            'D': data_loader.get_link_times()
        }
        
        # Variables
        X = m.addVars(links, vtype=GRB.BINARY, name="x")
        Y = m.addVars(V, vtype=GRB.BINARY, name="y")
        U = m.addVars(V, vtype=GRB.INTEGER, lb=1, ub=K, name="u")

        for key in links:
            if coeffs['C'][key] >= CRAZY_BUG_NUMBER:
                X[key].ub = 0

        # Objectives Expressions
        Z1_expr = X.prod(coeffs['C']) 
        Z2_expr = X.prod(coeffs['E']) 
        Z3_expr = Y.prod(R_i)         

        # Constraints
        m.addConstr(Y.sum('*') == K, "C1")
        m.addConstr(Y[S] == 1, "C2_Start")
        m.addConstr(Y[E] == 1, "C2_End")
        m.addConstrs((X.sum(i, '*', '*') == Y[i] for i in V if i != E), "C3_Out")
        m.addConstrs((X.sum('*', j, '*') == Y[j] for j in V if j != S), "C4_In")
        
        link_pairs = list(set((k[0], k[1]) for k in links))
        m.addConstrs((X.sum(i, j, '*') <= 1 for i, j in link_pairs), "C5_Mode")

        m.addConstr(X.sum('*', S, '*') == 0, "C6_Start")
        m.addConstr(X.sum(E, '*', '*') == 0, "C6_End")

        travel_time = X.prod(coeffs['D'])
        m.addConstr(travel_time + K * (T_race + T_rest_min) <= T_max, "C7_Time")

        for k in links:
            if k[2] == 'Sea':
                m.addConstr((coeffs['D'][k] - L_min) * X[k] >= 0, f"C8_Sea_{k}")

        n_nodes = len(V)
        for i in V:
            for j in V:
                if i != j:
                    relevant_links = X.sum(i, j, '*')
                    if (i,j) in link_pairs:
                        m.addConstr(U[i] - U[j] + n_nodes * relevant_links <= n_nodes - 1, f"C9_MTZ_{i}_{j}")

        m.addConstr(U[S] == 1, "C10_Start")
        m.addConstr(U[E] == K, "C10_End")

        # Objective Setup
        # Note: Revenue (Z3) is maximized, so we minimize Negative Revenue (-Z3)
        Obj_map = {1: Z1_expr, 2: Z2_expr, 3: -Z3_expr}

        if eps_values is None:
            m.setObjective(Obj_map[objective_index], GRB.MINIMIZE)
        else:
            epsilon2, epsilon3_prime = eps_values
            R2, R3_prime = R_ranges
            delta = 1e-6 
            S2 = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S2")
            S3_prime = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S3_prime")
            
            # Minimize Cost + Slacks
            m.setObjective(Z1_expr - delta * (S2/R2 + S3_prime/R3_prime), GRB.MINIMIZE)
            
            # Epsilon Constraints
            m.addConstr(Z2_expr + S2 == epsilon2, "Eps_Z2")
            m.addConstr((-Z3_expr) + S3_prime == epsilon3_prime, "Eps_Z3p")

        m.optimize()

        # --- EXTRACT ALL SOLUTIONS FROM POOL ---
        if m.status == GRB.OPTIMAL:
            # Iterate over all found solutions in the pool
            for k in range(m.SolCount):
                m.setParam(GRB.Param.SolutionNumber, k) # Select solution k
                
                # Reconstruct Path for solution k
                path_str = []
                curr = S
                # Simple path trace
                # Note: recovering the sequence from variables Xn is strictly needed
                # For visualization, we scan for active links
                active_links = []
                for key in links:
                    if X[key].Xn > 0.5: # Use Xn for pool solution value
                        active_links.append(key)
                
                # Since we have the active links, let's sort them by connectivity to make a route string
                # Simple ordered reconstruction
                curr_node = S
                for _ in range(K-1):
                    found = False
                    for (u_node, v_node, mode) in active_links:
                        if u_node == curr_node:
                            c_name = id_map.get(u_node, u_node)
                            j_name = id_map.get(v_node, v_node)
                            path_str.append(f"{c_name} -> {j_name} ({mode})")
                            curr_node = v_node
                            found = True
                            break
                    if not found: break
                
                full_route_text = "\n".join(path_str) if path_str else "Route Reconstruction Error"

                # Calculate objective values manually for this pool solution
                # (Gurobi objective value might include slack penalties)
                z1_val = sum(coeffs['C'][key] * X[key].Xn for key in links)
                z2_val = sum(coeffs['E'][key] * X[key].Xn for key in links)
                z3_val = sum(R_i.get(i,0) * Y[i].Xn for i in V)

                found_solutions.append({
                    'status': "Optimal",
                    'Z1': z1_val,
                    'Z2': z2_val,
                    'Z3': z3_val,
                    'route_list': path_str,
                    'route_text': full_route_text
                })
                
        return found_solutions

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        return []

# ==========================================
# 2. ADAPTIVE RECTANGULAR ALGORITHM
# ==========================================

def adaptive_epsilon_rectangular_method_gurobi(data_loader):
    print("--- 1. SINGLE OBJECTIVE OPTIMIZATIONS (EXTREMES) ---")
    
    # We allow the solver to return lists of solutions now
    sols1 = create_and_solve_gurobi_milp(data_loader, 1, 20) # Min Cost
    sols2 = create_and_solve_gurobi_milp(data_loader, 2, 20) # Min Emission
    sols3p = create_and_solve_gurobi_milp(data_loader, 3, 20) # Max Revenue

    if not sols1 or not sols2 or not sols3p:
        print("Error: Could not solve single objectives.")
        return []

    # Best extremes (Index 0 is the optimal one)
    best_cost = sols1[0]
    best_emit = sols2[0]
    best_rev = sols3p[0]

    print(f"\n[A] Best Cost: Cost=${best_cost['Z1']:,.0f}, Emit={best_cost['Z2']:.0f}, Rev=${best_cost['Z3']:,.0f}")
    print(f"[B] Best Emit: Cost=${best_emit['Z1']:,.0f}, Emit={best_emit['Z2']:.0f}, Rev=${best_emit['Z3']:,.0f}")
    print(f"[C] Best Rev:  Cost=${best_rev['Z1']:,.0f}, Emit={best_rev['Z2']:.0f}, Rev=${best_rev['Z3']:,.0f}")

    # --- COLLECT ALL UNIQUE SOLUTIONS ---
    # We dump ALL pool solutions into the candidate list
    # This ensures "near optimal" solutions are included in the final ranking
    NDS = []
    
    def add_candidates(sol_list, label_prefix):
        for idx, s in enumerate(sol_list):
            entry = {
                'label': f"{label_prefix}_{idx+1}",
                'cost': s['Z1'], 
                'emission': s['Z2'], 
                'revenue': s['Z3'], 
                'route_text': s['route_text']
            }
            # Check for duplicates based on objective values
            is_dupe = any(
                np.isclose(x['cost'], entry['cost']) and 
                np.isclose(x['emission'], entry['emission']) and
                np.isclose(x['revenue'], entry['revenue'])
                for x in NDS
            )
            if not is_dupe:
                NDS.append(entry)

    add_candidates(sols1, "MinCost")
    add_candidates(sols2, "MinEmit")
    add_candidates(sols3p, "MaxRev")

    # --- PARETO GENERATION ---
    print("\n--- 2. ADAPTIVE RECTANGULAR SEARCH (PARETO GENERATION) ---")
    
    # Ranges for epsilon method
    z2_vals = [s['Z2'] for s in [best_cost, best_emit, best_rev]]
    z3p_vals = [-s['Z3'] for s in [best_cost, best_emit, best_rev]]
    
    R2 = max(z2_vals) - min(z2_vals) + 1
    R3p = max(z3p_vals) - min(z3p_vals) + 1
    R_ranges = (R2, R3p)
    
    z2_min, z2_max = min(z2_vals), max(z2_vals)
    z3p_min, z3p_max = min(z3p_vals), max(z3p_vals)
    
    rectangles = deque([(z2_min, z2_max, z3p_min, z3p_max)])
    
    count = 0
    while rectangles and count < 15: 
        rec = rectangles.popleft()
        eps2, eps3p = rec[1], rec[3]
        
        # Get list of solutions from subproblem
        sub_sols = create_and_solve_gurobi_milp(data_loader, 1, 10, (eps2, eps3p), R_ranges)
        
        if sub_sols:
            # We only use the BEST solution for the geometric splitting logic
            best_res = sub_sols[0]
            
            # But we add ALL solutions found to our result pool
            add_candidates(sub_sols, f"TradeOff_{count}")
            
            curr_z2 = best_res['Z2']
            curr_z3p = -best_res['Z3']
            
            # Simple check to avoid infinite loops if solver returns same point
            if curr_z2 > rec[0] and curr_z3p > rec[2]:
                if curr_z2 < rec[1]: # Only split if inside bounds
                    rectangles.append((rec[0], curr_z2, rec[2], rec[3]))
                if curr_z3p < rec[3]:
                    rectangles.append((curr_z2, rec[1], rec[2], curr_z3p))
                count += 1
    
    return NDS

# ==========================================
# 3. MCDM (RANKING)
# ==========================================

def perform_ranking(pareto_list):
    print("\n--- 3. MCDM RANKING (D-CRITIC + TOPSIS) ---")
    df = pd.DataFrame(pareto_list)
    
    # Normalize
    norm = pd.DataFrame()
    for col in ['cost', 'emission']: 
        denom = df[col].max() - df[col].min()
        if denom == 0: denom = 1
        norm[col] = (df[col].max() - df[col]) / denom # Min
        
    denom = df['revenue'].max() - df['revenue'].min()
    if denom == 0: denom = 1
    norm['revenue'] = (df['revenue'] - df['revenue'].min()) / denom # Max
    
    # Weights
    std = norm.std().fillna(0)
    corr = norm.corr().fillna(0)
    
    raw_weights = {}
    total = 0
    for j in norm.columns:
        conflict = sum([1 - corr[j][k] for k in norm.columns])
        val = std[j] * conflict
        raw_weights[j] = val
        total += val
    
    if total == 0:
        final_w = {k: 0.33 for k in raw_weights}
    else:
        final_w = {k: v/total for k,v in raw_weights.items()}
        
    print(f"Objective Weights: {final_w}")
    
    # TOPSIS
    vec = pd.DataFrame()
    for col in ['cost','emission','revenue']:
        rss = np.sqrt((df[col]**2).sum())
        if rss == 0: rss = 1
        vec[col] = (df[col] / rss) * final_w[col]
        
    pis = {
        'cost': vec['cost'].min(), 'emission': vec['emission'].min(), 'revenue': vec['revenue'].max()
    }
    nis = {
        'cost': vec['cost'].max(), 'emission': vec['emission'].max(), 'revenue': vec['revenue'].min()
    }
    
    d_pis = np.sqrt(((vec[['cost','emission','revenue']] - pd.Series(pis))**2).sum(axis=1))
    d_nis = np.sqrt(((vec[['cost','emission','revenue']] - pd.Series(nis))**2).sum(axis=1))
    
    denom = d_pis + d_nis
    df['closeness'] = d_nis / denom.replace(0, 1)
    df['rank'] = df['closeness'].rank(ascending=False)
    
    # Final Output
    df = df.sort_values('rank')
    
    print(f"\n--- FINAL RANKINGS (Showing top {min(len(df), 5)} of {len(df)} candidates) ---")
    
    top_n = df.head(5)
    for idx, row in top_n.iterrows():
        print(f"\nRANK {int(row['rank'])}: {row['label']}")
        print(f"  Score: {row['closeness']:.4f}")
        print(f"  Stats: Cost=${row['cost']:,.0f}, Emit={row['emission']:.0f}, Rev=${row['revenue']:,.0f}")
        print(f"  Route:")
        print("    " + row['route_text'].replace("\n", "\n    "))

if __name__ == "__main__":
    loader = DataLoader()
    if loader.circuits:
        results = adaptive_epsilon_rectangular_method_gurobi(loader)
        if len(results) > 0:
            perform_ranking(results)
        else:
            print("No solutions found.")
    else:
        print("System Error: No circuits loaded.")