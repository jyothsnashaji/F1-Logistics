import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from collections import deque
import math
from data_loader import DataLoader
from constants import *
# --- Core Optimization Function (Gurobi) ---

def create_and_solve_gurobi_milp(data_loader, objective_index, eps_values=None, R_ranges=None):
    """
    Creates and solves the single-objective MILP (AUGMECON) using Gurobi.
    Objective index: 1 (Z1: Cost), 2 (Z2: Emissions), 3 (Z3': Negative Revenue)
    """
    try:
        # Load Parameters
        m = gp.Model("F1_Logistics_MOO_Subproblem")
        m.setParam('OutputFlag', 0) # Suppress Gurobi output

        V = data_loader.load_circuits()
        M = ['Air', 'Road', 'Sea']
        S, E = data_loader.get_season_start_end()
        K = data_loader.get_total_num_required_races()
        T_max = data_loader.get_max_season_days()
        T_race = data_loader.get_num_days_per_race()
        T_rest_min = data_loader.get_minimum_rest_days()
        L_min = data_loader.get_lead_time_min()
        
        # Coefficients and decision-variable domain restricted to actual transport links
        links = data_loader.keys
        R_i = data_loader.get_circuit_revenue()

        coeffs = {
            'C': data_loader.get_link_costs(), # Cost
            'E': data_loader.get_link_emissions(), # Emission
            'D': data_loader.get_link_times() # Time (uniform per link)
        }
        
        # 1. Decision Variables
        X = m.addVars(links, vtype=GRB.BINARY, name="x")      # x_ij^m
        Y = m.addVars(V, vtype=GRB.BINARY, name="y")          # y_i
        U = m.addVars(V, vtype=GRB.INTEGER, lb=1, ub=K, name="u") # u_i (Sequence position index, C12)

        for key in links:
            cost_map = data_loader.get_link_costs() 
            if cost_map[key] == CRAZY_BUG_NUMBER:
                X[key].ub = 0

        # 2. Objective Functions (Expressions)
        # Z1: Total Logistical Cost (Minimize)
        Z1 = X.prod(coeffs['C']) 
        # Z2: Total Carbon Emissions (Minimize)
        Z2 = X.prod(coeffs['E'])
        # Z3: Total Commercial Revenue (Maximize)
        Z3 = Y.prod(R_i)
        
        # Z3' (Negative Revenue) for minimization
        Z3_prime = -Z3

        Obj_map = {1: Z1, 2: Z2, 3: Z3_prime}

        # 3. Constraints (C1 - C10)
        
        # C1: Race Count: Sum(y_i) = K
        m.addConstr(Y.sum('*') == K, name="C1_Race_Count")
        
        # C2: Fixed Start/End: y_S = 1, y_E = 1
        m.addConstr(Y[S] == 1, name="C2_Start_Fixed")
        m.addConstr(Y[E] == 1, name="C2_End_Fixed")
            
        # C3: Single Link Out: Sum_j,m (x_ij^m) = y_i for all i
        m.addConstrs((X.sum(i, '*', '*') == Y[i] for i in V if i != E), name="C3_Link_Out_i_ne_E")

        # C4: Single Link In: Sum_i,m (x_ij^m) = y_j for all j
        m.addConstrs((X.sum('*', j, '*') == Y[j] for j in V if j != S), name="C4_Link_In_j_ne_S")
        
        # C5: Single Mode Per Link: Sum_m (x_ij^m) <= 1 for all (i,j) present in transport data
        links_ij = sorted(set((from_id, to_id) for from_id, to_id, _ in links))
        m.addConstrs((X.sum(i, j, '*') <= 1 for i, j in links_ij), name="C5_Single_Mode")

        # C6: Terminal Flow: No flow into_id S, No flow out of E
        m.addConstr(X.sum('*', S, '*') == 0, name="C6_Flow_In_S")
        m.addConstr(X.sum(E, '*', '*') == 0, name="C6_Flow_Out_E")

        # C7: To_idtal Time Budget: Sum(D_ij^m * x_ij^m) + K * (T_race + T_rest_min) <= T_max
        travel_time = X.prod(coeffs['D'])
        to_idtal_fixed_time = K * (T_race + T_rest_min)
        m.addConstr(travel_time + to_idtal_fixed_time <= T_max, name="C7_To_idtal_Time_Budget")

        # C8: Sea Freight Lead Time: D_ij^Sea * x_ij^Sea >= L_min * x_ij^Sea
        # Note: This is equivalent to_id D_ij^Sea >= L_min ONLY if x_ij^Sea = 1.
        # This constraint is inherently satisfied if the input data D_ij^Sea >= L_min.
        # However, to_id be mathematically rigorous: (D_ij^Sea - L_min) * x_ij^Sea >= 0
        for i, j in links_ij:
            # Apply lead-time check only if a Sea-mode link exists in data
            if (i, j, 'Sea') in coeffs['D']:
                D_sea = coeffs['D'][i, j, 'Sea']
                m.addConstr((D_sea - L_min) * X[i, j, 'Sea'] >= 0, name=f"C8_Sea_Lead_{i}_{j}")

        # C9: Subto_idur Elimination (MTZ): u_i - u_j + K * Sum_m (x_ij^m) <= K - 1
        for i in V:
            for j in V:
                if i != j:
                    m.addConstr(U[i] - U[j] + K * X.sum(i, j, '*') <= K - 1, name=f"C9_MTZ_{i}_{j}")

        # C10: Sequence Fixed Points: u_S = 1, u_E = K
        m.addConstr(U[S] == 1, name="C10_Seq_Start")
        m.addConstr(U[E] == K, name="C10_Seq_End")

        # C11 & C12: Variable domains (set during definition)
        # We also need to_id enforce U[i] to_id be set only if Y[i]=1, but since 
        # the to_idur constraints (C3, C4) ensure that only selected nodes are part of the path, 
        # and C9 links X to_id U, the variable U will only have meaningful values 
        # for selected nodes. Since U is constrained to_id [1, K], this is usually sufficient.
        
        # 4. Set Objective and Epsilon Constraints
        if eps_values is None:
            # Phase 1: Solving for Ideal/Nadir points
            m.setObjective(Obj_map[objective_index], GRB.MINIMIZE)
        else:
            # Phase 2: Solving the AUGMECON sub-problem
            epsilon2, epsilon3_prime = eps_values
            R2, R3_prime = R_ranges
            delta = 1e-6 
            
            # Slack variables for the epsilon constraints
            S2 = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S2") # Slack for Z2 (Emissions)
            S3_prime = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S3_prime") # Slack for Z3' (Neg. Revenue)
            
            # Primary Objective: Z1 - delta * (S2/R2 + S3'/R3') -> MINIMIZE
            m.setObjective(Z1 - delta * (S2/R2 + S3_prime/R3_prime), GRB.MINIMIZE)
            
            # Epsilon Constraints (Z_k + S_k = epsilon_k)
            # Z2 (Emissions) <= epsilon2
            m.addConstr(Z2 + S2 == epsilon2, "Epsilon_Constraint_Z2")
            # Z3' (Neg. Revenue) <= epsilon3_prime
            m.addConstr(Z3_prime + S3_prime == epsilon3_prime, "Epsilon_Constraint_Z3_prime")

        # 5. Optimize
        m.optimize()

        if m.status == GRB.OPTIMAL:
            # Collect results
            m.update()
            # --- Objective Values ---
            Z1_val = Z1.getValue()
            Z2_val = Z2.getValue()
            Z3_val = Z3.getValue()
            
            # --- Decision Variables ---
            
            # 1. Selected Circuits (y_i)
            selected_circuits = [i for i in V if Y[i].X > 0.5]
            
            # 2. Selected Links (x_ij^m)
            selected_links = []
            for (i, j, mode) in links:
                if X[i, j, mode].x > 0.5:
                    selected_links.append({
                        'from_id': i, 
                        'to_id': j, 
                        'mode': mode, 
                        'sequence_start': U[i].X if i in selected_circuits else None # Add sequence index for clarity
                    })
            
            # Sort the links by the sequence index (u_i) for a readable to_idur
            selected_links.sort(key=lambda link: link.get('sequence_start', float('inf')))

            # Collect results
            results = {
                'status': "Optimal",
                'Z1': Z1_val,
                'Z2': Z2_val,
                'Z3': Z3_val,
                'selected_circuits': selected_circuits,
                'selected_links': selected_links
            }
            return results
        if m.status == GRB.INFEASIBLE:
            print("\n--- Gurobi Status 3: MODEL INFEASIBLE ---")
            
            # 1. Compute IIS (Irreducible Inconsistent Subsystem)
            m.computeIIS()
            print("The following is an Irreducible Inconsistent Subsystem (IIS):")
            
            infeasible_constraints = []
            
            # 2. Iterate through all constraints and print those in the IIS
            for c in m.getConstrs():
                if c.IISConstr: # If the constraint is part of the IIS
                    infeasible_constraints.append(f"Hard Constraint: {c.ConstrName} (RHS: {c.RHS})")
                
            # 3. For the AUGMECON phase, check the objective constraints to_ido
            # These are the constraints C_eps_Z2 and C_eps_Z3_prime
            if eps_values is not None:
                for c in m.getConstrs():
                    if c.ConstrName.startswith("Epsilon_Constraint") and c.IISConstr:
                         infeasible_constraints.append(f"Epsilon Constraint: {c.ConstrName} (Value: {c.RHS:.2f})")
                         
            # 4. Print the final list
            if infeasible_constraints:
                print("\n".join(infeasible_constraints))
            else:
                print("IIS found no constraints. (Check if variables or bounds are conflicting instead).")
                
            return {'status': "Infeasible - IIS generated"}
        return {'status': f"Gurobi Status: {m.status}"}

    except gp.GurobiError as e:
        return {'status': f"Gurobi Error: {e}"}
    except Exception as e:
        return {'status': f"General Error: {e}"}

# -------------------------------------------------------------
# --- Main Adaptive Epsilon Algorithm (Gurobi Integration) ---
# -------------------------------------------------------------

def adaptive_epsilon_rectangular_method_gurobi(data_loader):
    """
    Implements the full Adaptive Epsilon (Rectangular) Method.
    Constrained objectives are Z2 (Emissions, Min) and Z3' (Neg. Revenue, Min).
    """
    
    print("--- Phase 1: Initialization (Ideal & Nadir Points) ---")
    
    # 1. Compute Payoff Table
    
    # Objective indices: 1 (Z1: Cost, MIN), 2 (Z2: Emissions, MIN), 3 (Z3': Neg. Revenue, MIN)
    sol1 = create_and_solve_gurobi_milp(data_loader, objective_index=1)
    sol2 = create_and_solve_gurobi_milp(data_loader, objective_index=2)
    # Z3 is MAX Revenue, so we optimize Z3' (Neg. Revenue) for MIN
    sol3_prime = create_and_solve_gurobi_milp(data_loader, objective_index=3)

    if not all(sol['status'] == 'Optimal' for sol in [sol1, sol2, sol3_prime]):
        return "Error: Initial single-objective problems could not be solved to_id optimality."

    # Z_matrix rows: [Z1, Z2, Z3] (Z3 is Max Revenue)
    Z_matrix = np.array([
        [sol1['Z1'], sol2['Z1'], sol3_prime['Z1']], 
        [sol1['Z2'], sol2['Z2'], sol3_prime['Z2']], 
        [sol1['Z3'], sol2['Z3'], sol3_prime['Z3']], 
    ])
    
    # Ideal Point: (min Z1, min Z2, max Z3)
    Z_ideal = (np.min(Z_matrix[0, :]), np.min(Z_matrix[1, :]), np.max(Z_matrix[2, :]))
    
    # Nadir Point: (max Z1, max Z2, min Z3)
    Z_nadir = (np.max(Z_matrix[0, :]), np.max(Z_matrix[1, :]), np.min(Z_matrix[2, :]))
    
    print(f"Ideal Point (Z1 min, Z2 min, Z3 max): ({Z_ideal[0]:.2f}, {Z_ideal[1]:.2f}, {Z_ideal[2]:.2f})")
    print(f"Nadir Point (Z1 max, Z2 max, Z3 min): ({Z_nadir[0]:.2f}, {Z_nadir[1]:.2f}, {Z_nadir[2]:.2f})")

    # Calculate Ranges for constrained objectives (Z2 and Z3')
    R_Z2 = Z_nadir[1] - Z_ideal[1] # Range for Emissions (Z2)
    R_Z3 = Z_ideal[2] - Z_nadir[2] # Range for Revenue (Z3)
    
    # Convert Z3 values (Max Revenue) to_id Z3' (Min Neg. Revenue)
    Z3_prime_min = -Z_ideal[2] 
    Z3_prime_max = -Z_nadir[2] 
    R_Z3_prime = Z3_prime_max - Z3_prime_min # Range is the same as R_Z3

    R_ranges = (R_Z2, R_Z3_prime)
    
    # Initial Non-Dominated Set (NDS) - Include Z1-optimal solution
    NDS = {(sol1['Z1'], sol1['Z2'], sol1['Z3'])}
    
    # Initial Rectangle (in Z2 and Z3' space): [Z2_min, Z2_max] x [Z3'_min, Z3'_max]
    initial_rectangle = (Z_ideal[1], Z_nadir[1], Z3_prime_min, Z3_prime_max)
    Rectangles = deque([initial_rectangle])

    print("\n--- Phase 2: Adaptive Epsilon (Rectangular Search) ---")
    
    # 2. Rectangular Search Loop
    while Rectangles:
        eps2_low, eps2_high, eps3p_low, eps3p_high = Rectangles.popleft()
        
        # Set the Epsilon-Constraints (Use the 'high' values for min-min problems)
        epsilon2 = eps2_high         # Emissions constraint
        epsilon3_prime = eps3p_high  # Negative Revenue constraint
        
        # Solve the AUGMECON Sub-Problem
        new_sol = create_and_solve_gurobi_milp(
            data_loader, 
            objective_index=1, 
            eps_values=(epsilon2, epsilon3_prime), 
            R_ranges=R_ranges
        )
        
        if new_sol['status'] != 'Optimal':
            continue
            
        Z_new = (new_sol['Z1'], new_sol['Z2'], new_sol['Z3'])
        
        # Non-Dominance Check and Update NDS (Ensures solution is unique)
        if Z_new not in NDS:
            NDS.add(Z_new)
            
            # Convert Z_new values for splitting the rectangles
            Z_new2 = Z_new[1]         # Emissions
            Z_new3_prime = -Z_new[2]  # Negative Revenue

            # Decomposition (Rectangular Splitting)
            
            # New Rectangle 1: Tighter Constraint on Z2 (Emissions)
            # [Z2_low, Z2_new] x [Z3'_low, Z3'_high]
            if Z_new2 > eps2_low and Z_new2 < eps2_high:
                 new_rect1 = (eps2_low, Z_new2, eps3p_low, eps3p_high)
                 Rectangles.append(new_rect1)
                 
            # New Rectangle 2: Tighter Constraint on Z3' (Negative Revenue)
            # [Z2_new, Z2_high] x [Z3'_low, Z3'_new]
            if Z_new3_prime > eps3p_low and Z_new3_prime < eps3p_high:
                new_rect2 = (Z_new2, eps2_high, eps3p_low, Z_new3_prime)
                Rectangles.append(new_rect2)
                
    # 3. Final Results
    NDS_list = sorted(list(NDS))
    return NDS_list

# --- Execute the Code ---

try:
    loader = DataLoader() 
    pareto_id_front = adaptive_epsilon_rectangular_method_gurobi(loader)
    
    print("\n--- Phase 3: Final Non-Dominated Solutions (Pareto_id Front) ---")
    print(f"Found {len(pareto_id_front)} Non-Dominated Solution(s).")
    print("Format: (Cost, Revenue, Emission)")
    # 
    
    # Output table of non-dominated solutions
    table = []
    for i, (z1, z2, z3) in enumerate(pareto_id_front):
         table.append(f"Solution {i+1}: Cost={z1:.2f}, Revenue={z2:.2f}, Emission={z3:.2f}")

    print("\n".join(table))

except Exception as e:
    print(f"An error occurred during execution: {e}")