import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from collections import deque
import math
from data_loader import DataLoader

# --- Core Optimization Function (Gurobi) ---

def create_and_solve_gurobi_milp(data_loader, objective_index, eps_values=None, R_ranges=None):
    """
    Creates and solves the single-objective MILP using Gurobi.
    """
    try:
        # Model setup
        m = gp.Model("F1_Logistics_MOO_Subproblem")
        m.setParam('OutputFlag', 0) # Suppress Gurobi output for cleaner execution

        C = data_loader.load_circuits()
        M = ['Air', 'Road', 'Sea']
        start, end = data_loader.get_season_start_end()
        
        # Define key set
        links = [(i, j, m) for i in C for j in C if i != j for m in M]

        # Extract coefficients
        coeffs = {
            'cost': {(l['from'], l['to'], l['mode']): l['cost'] for l in data_loader.load_transport_links()},
            'neg_revenue': {(l['from'], l['to'], l['mode']): l['neg_revenue'] for l in data_loader.load_transport_links()},
            'emission': {(l['from'], l['to'], l['mode']): l['emission'] for l in data_loader.load_transport_links()}
        }

        # 1. Decision Variables
        # Binary variable X[i, j, m] is 1 if link (i, j) with mode m is used
        X = m.addVars(links, vtype=GRB.BINARY, name="X") 
        # Integer variable T[c] is the sequence index for circuit c
        T = m.addVars(C, vtype=GRB.INTEGER, lb=1, ub=len(C), name="T")

        # 2. Objective Functions (Expressions)
        Z1 = X.prod(coeffs['cost']) # Cost (Min)
        Z2_prime = X.prod(coeffs['neg_revenue']) # Negative Revenue (Min)
        Z3 = X.prod(coeffs['emission']) # Emission (Min)

        Obj_map = {1: Z1, 2: Z2_prime, 3: Z3}

        # 3. Standard Constraints (TSP and Logistics)
        N = len(C)
        M_big = N 

        # Tour Constraints: Enter and Exit each circuit exactly once
        m.addConstrs((X.sum(i, '*', '*') == 1 for i in C), name="Outflow")
        m.addConstrs((X.sum('*', i, '*') == 1 for i in C), name="Inflow")
            
        # Subtour Elimination Constraints (MTZ formulation)
        for i in C:
            for j in C:
                if i != j and i != start:
                    # T[i] - T[j] + M * X_sum_ij <= M - 1
                    m.addConstr(T[i] - T[j] + M_big * X.sum(i, j, '*') <= M_big - 1, f"MTZ_{i}_{j}")

        # 4. Set Objective and Epsilon Constraints
        if eps_values is None:
            # Phase 1: Solving for Ideal/Nadir points
            m.setObjective(Obj_map[objective_index], GRB.MINIMIZE)
        else:
            # Phase 2: Solving the AUGMECON sub-problem
            epsilon2, epsilon3 = eps_values
            R2, R3 = R_ranges
            delta = 1e-6 # Small perturbation for strict non-dominance
            
            # Slack variables for the epsilon constraints
            S2 = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S2")
            S3 = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="S3")
            
            # Primary Objective: Z1 (Cost) plus augmentation term
            # Note: Gurobi expressions handle division by constants correctly
            m.setObjective(Z1 - delta * (S2/R2 + S3/R3), GRB.MINIMIZE)
            
            # Epsilon Constraints (Z' + S = epsilon)
            # Z2' (Neg. Revenue) <= epsilon2
            m.addConstr(Z2_prime + S2 == epsilon2, "Epsilon_Constraint_Z2")
            # Z3 (Emission) <= epsilon3
            m.addConstr(Z3 + S3 == epsilon3, "Epsilon_Constraint_Z3")

        # 5. Optimize
        m.optimize()

        if m.status == GRB.OPTIMAL:
            # Collect results
            results = {
                'status': "Optimal",
                'Z1': Z1.getValue(),
                'Z2': -Z2_prime.getValue(), # Convert back to Max Revenue
                'Z3': Z3.getValue(),
            }
            return results
        
        return {'status': f"Gurobi Status: {m.status}"}

    except gp.GurobiError as e:
        return {'status': f"Gurobi Error: {e}"}
    except Exception as e:
        return {'status': f"General Error: {e}"}

# --- Main Adaptive Epsilon Algorithm (Gurobi Integration) ---

def adaptive_epsilon_rectangular_method_gurobi(data_loader):
    """
    Implements the full Adaptive Epsilon (Rectangular) Method using Gurobi.
    """
    
    print("--- Phase 1: Initialization (Ideal & Nadir Points) ---")
    
    # 1. Compute Payoff Table (Ideal Point and Nadir Bounds)
    
    # 1: Min Cost (Primary), 2: Min Neg. Revenue (Max Revenue), 3: Min Emission
    sol1 = create_and_solve_gurobi_milp(data_loader, objective_index=1)
    sol2 = create_and_solve_gurobi_milp(data_loader, objective_index=2)
    sol3 = create_and_solve_gurobi_milp(data_loader, objective_index=3)

    if not all(sol['status'] == 'Optimal' for sol in [sol1, sol2, sol3]):
        # Handle cases where initial solutions are infeasible/error
        print("Error: Initial single-objective problems could not be solved to optimality.")
        return [sol1, sol2, sol3]

    # Z_matrix rows: [Z1, Z2, Z3]
    Z_matrix = np.array([
        [sol1['Z1'], sol2['Z1'], sol3['Z1']], 
        [sol1['Z2'], sol2['Z2'], sol3['Z2']], 
        [sol1['Z3'], sol2['Z3'], sol3['Z3']], 
    ])
    
    # Ideal Point: (min Z1, max Z2, min Z3)
    Z_ideal = (np.min(Z_matrix[0, :]), np.max(Z_matrix[1, :]), np.min(Z_matrix[2, :]))
    
    # Nadir Point: (max Z1, min Z2, max Z3)
    Z_nadir = (np.max(Z_matrix[0, :]), np.min(Z_matrix[1, :]), np.max(Z_matrix[2, :]))
    
    print(f"Ideal Point (Z1 min, Z2 max, Z3 min): ({Z_ideal[0]:.2f}, {Z_ideal[1]:.2f}, {Z_ideal[2]:.2f})")
    print(f"Nadir Point (Z1 max, Z2 min, Z3 max): ({Z_nadir[0]:.2f}, {Z_nadir[1]:.2f}, {Z_nadir[2]:.2f})")

    # Calculate Ranges for constrained objectives (Z2' and Z3)
    R_Z2 = Z_nadir[1] - Z_ideal[1]
    R_Z3 = Z_nadir[2] - Z_ideal[2]
    
    # Convert Z2 values (Revenue) to Z2' (Neg. Revenue)
    Z2_prime_min = -Z_ideal[1] 
    Z2_prime_max = -Z_nadir[1] 
    R_Z2_prime = R_Z2 # The range magnitude is the same

    R_ranges_prime = (R_Z2_prime, R_Z3)
    
    # Initial Non-Dominated Set (NDS)
    NDS = {(sol1['Z1'], sol1['Z2'], sol1['Z3'])}
    
    # Initial Rectangle (in Z2' and Z3 space): [Z2'_min, Z2'_max] x [Z3_min, Z3_max]
    initial_rectangle = (Z2_prime_min, Z2_prime_max, Z_ideal[2], Z_nadir[2])
    Rectangles = deque([initial_rectangle])

    print("\n--- Phase 2: Adaptive Epsilon (Rectangular Search) ---")
    
    # 2. Rectangular Search Loop
    while Rectangles:
        eps2_low, eps2_high, eps3_low, eps3_high = Rectangles.popleft()
        
        # Set the Epsilon-Constraints (Use the 'high' values for min-min problems)
        epsilon2 = eps2_high 
        epsilon3 = eps3_high
        
        print(f"\nSolving P($\epsilon_2$={epsilon2:.2f}, $\epsilon_3$={epsilon3:.2f})")
        
        # Solve the AUGMECON Sub-Problem
        new_sol = create_and_solve_gurobi_milp(
            data_loader, 
            objective_index=1, 
            eps_values=(epsilon2, epsilon3), 
            R_ranges=R_ranges_prime
        )
        
        if new_sol['status'] != 'Optimal':
            print(f"Rectangle for $\epsilon=({epsilon2:.2f}, {epsilon3:.2f})$ is infeasible or error. Fathomed.")
            continue
            
        Z_new = (new_sol['Z1'], new_sol['Z2'], new_sol['Z3'])
        
        # Non-Dominance Check and Update NDS
        # This basic check ensures uniqueness, as AUGMECON should find only non-dominated points
        if Z_new not in NDS:
            NDS.add(Z_new)
            print(f"-> Found NEW Non-Dominated Solution: Z=({Z_new[0]:.2f}, {Z_new[1]:.2f}, {Z_new[2]:.2f})")
            
            # Convert Z_new[1] (Max Revenue) to Z_new_prime (Min Neg. Revenue)
            Z_new_prime2 = -Z_new[1]
            Z_new_prime3 = Z_new[2]

            # Decomposition (Rectangular Splitting)
            
            # New Rectangle 1: Tighter Constraint on Z2' (Neg. Revenue)
            if Z_new_prime2 > eps2_low and Z_new_prime2 < eps2_high:
                 new_rect1 = (eps2_low, Z_new_prime2, eps3_low, eps3_high)
                 Rectangles.append(new_rect1)
                 
            # New Rectangle 2: Tighter Constraint on Z3 (Emission)
            if Z_new_prime3 > eps3_low and Z_new_prime3 < eps3_high:
                new_rect2 = (Z_new_prime2, eps2_high, eps3_low, Z_new_prime3)
                Rectangles.append(new_rect2)
                
    # 3. Final Results
    NDS_list = sorted(list(NDS))
    return NDS_list

# --- Execute the Code ---

try:
    # Set up data loader (e.g., 5 circuits)
    loader = DataLoader(num_circuits=5) 
    pareto_front = adaptive_epsilon_rectangular_method_gurobi(loader)
    
    print("\n--- Phase 3: Final Non-Dominated Solutions (Pareto Front) ---")
    print(f"Found {len(pareto_front)} Non-Dominated Solution(s).")
    print("Format: (Cost, Revenue, Emission)")
    # 
    
    # Output table of non-dominated solutions
    table = []
    for i, (z1, z2, z3) in enumerate(pareto_front):
         table.append(f"Solution {i+1}: Cost={z1:.2f}, Revenue={z2:.2f}, Emission={z3:.2f}")

    print("\n".join(table))

except Exception as e:
    print(f"An error occurred during execution: {e}")