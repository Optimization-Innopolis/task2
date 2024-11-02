import numpy as np

def simplex_solver(C, A, b, eps):
    
    print("Optimization Problem:")
    objective_type = "Maximize"
    obj_func = " + ".join(f"{C[i]} * x{i + 1}" for i in range(len(C)))
    print(f"{objective_type} z = {obj_func}")

    print("Subject to the constraints:")
    for i in range(len(A)):
        constraints = " + ".join(f"{A[i][j]} * x{j + 1}" for j in range(len(A[i])))
        print(f"{constraints} <= {b[i]}")

    num_vars = len(C)
    num_constraints = len(A)

    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))

    tableau[:num_constraints, :num_vars] = A
    tableau[:num_constraints, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    tableau[:num_constraints, -1] = b
    tableau[-1, :num_vars] = -C

    while True:
        entering_var_index = np.argmin(tableau[-1, :-1])
        entering_var_coeff = tableau[-1, entering_var_index]

        if entering_var_coeff >= -eps:
            break

        ratios = []
        for i in range(num_constraints):
            if tableau[i, entering_var_index] > eps:
                ratios.append(tableau[i, -1] / tableau[i, entering_var_index])
            else:
                ratios.append(float('inf'))

        leaving_var_index = np.argmin(ratios)

        if ratios[leaving_var_index] == float('inf'):
            return {"solver_state": "Method is not applicable!", "x_star": None, "z": None}

        pivot_value = tableau[leaving_var_index, entering_var_index]
        tableau[leaving_var_index] /= pivot_value

        for i in range(num_constraints + 1):
            if i != leaving_var_index:
                tableau[i] -= tableau[i, entering_var_index] * tableau[leaving_var_index]

    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[:, i]
        if sum(col == 1) == 1 and sum(col == 0) == num_constraints:
            row = np.where(col == 1)[0][0]
            solution[i] = tableau[row, -1]

    optimal_value = tableau[-1, -1]

    return {"solver_state": "solved", "x_star": solution, "z": optimal_value}

def interior_point_method(C, A, eps, alpha, x_initial):
    # Initialize decision variables with defined starting point
    x = x_initial
    n = len(C)
    I = np.eye(n)
    
    iteration = 0
    while True:
        # Save current solution to check for convergence later
        x_prev = x.copy()
        
        # Step 1: Create diagonal matrix D from current x
        D = np.diag(x)
        
        # Step 2: Calculate modified matrices
        A_hat = A @ D
        c_hat = D @ C
        
        # Step 3: Calculate projection matrix P
        try:
            F = A_hat @ A_hat.T
            F_inv = np.linalg.inv(F)
            H = A_hat.T @ F_inv @ A_hat
            P = I - H
        except np.linalg.LinAlgError:
            return "The method is not applicable!"  # Handle singular matrix case
        
        # Step 4: Calculate projected cost vector
        c_p = P @ c_hat
        
        # Check if solution is feasible
        if all(cp >= 0 for cp in c_p):
            return "The problem does not have a solution!"
        
        # Step 5: Identify the largest negative component of c_p for update step
        nu = abs(min(c_p))  # max negative component in c_p
        
        # Step 6: Update decision variables
        x_hat = np.ones(n) + (alpha / nu) * c_p
        x = D @ x_hat
        
        # Increment iteration
        iteration += 1
        # Check for convergence
        if np.linalg.norm(x - x_prev, ord=2) < eps:
            break
    
    # Calculate objective function value
    optimal_value = C @ x
    return x, optimal_value, iteration

# Values for initial problem

# The approximation accuracy
eps = 0.001

# LPP Problem
# Input format for Simplex Method
Cs = np.array([3, 2])
As = np.array([[2, 1], [1, 1], [1, 0]])
bs = np.array([10, 8, 4])

# Input format for Interior-Point algorithm
C = np.array([3, 2, 0, 0, 0])
A = np.array([[2, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
b = np.array([10, 8, 4])

# Initial starting point
x_initial = np.array([1, 1, 7, 6, 3])

# Test 1
# Input format for Simplex Method
# Cs = np.array([5, 4])
# As = np.array([[1, 2], [3, 2]])
# bs = np.array([6, 12])

# # Input format for Interior-Point algorithm
# C = np.array([5, 4, 0, 0])
# A = np.array([[1, 2, 1, 0], [3, 2, 0, 1]])
# b = np.array([6, 12])

# # Initial starting point
# x_initial = np.array([1, 1, 3, 7])

# Test 2
# Input format for Simplex Method
# Cs = np.array([6, 8])
# As = np.array([[1, 1], [5, 4]])
# bs = np.array([10, 40])

# # Input format for Interior-Point algorithm
# C = np.array([6, 8, 0, 0])
# A = np.array([[1, 1, 1, 0], [5, 4, 0, 1]])
# b = np.array([10, 40])

# # Initial starting point
# x_initial = np.array([1, 1, 10, 31])

# Test 3
# Input format for Simplex Method
# Cs = np.array([4, 3])
# As = np.array([[-1, 1]])
# bs = np.array([2])

# # Input format for Interior-Point algorithm
# C = np.array([4, 3, 0])
# A = np.array([[-1, 1, 1]])
# b = np.array([2])

# # Initial starting point
# x_initial = np.array([1, 1, 2])

# Test 4
# Input format for Simplex Method
# Cs = np.array([10, 6])
# As = np.array([[1, 1], [2, 1], [1, 2]])
# bs = np.array([100, 150, 120])

# # Input format for Interior-Point algorithm
# C = np.array([10, 6, 0, 0, 0])
# A = np.array([[1, 1, 1, 0, 0], [2, 1, 0, 1, 0], [1, 2, 0, 0, 1]])
# b = np.array([100, 150, 120])

# # Initial starting point
# x_initial = np.array([1, 1, 98, 147, 117])

# Test 5
# Input format for Simplex Method
# Cs = np.array([2, 5])
# As = np.array([[1, 2], [2, 1]])
# bs = np.array([20, 18])

# # Input format for Interior-Point algorithm
# C = np.array([2, 5, 0, 0])
# A = np.array([[1, 2, 1, 0], [2, 1, 0, 1]])
# b = np.array([20, 18])

# # Initial starting point
# x_initial = np.array([1, 1, 17, 15])

result = simplex_solver(Cs, As, bs, eps)
print("\nResult by Simplex:")
print("Solver State:", result['solver_state'])
if result['solver_state'] == "solved":
    print("Optimal Decision Variables x*:", result['x_star'])
    print("Optimal Value:", result['z'])

print("\nInterior-Point Method:")
print("Initial Point:", x_initial)

print("\nResults by Interior-Point Method:")
# Run with alpha = 0.5
result_a1 = interior_point_method(C, A, eps, 0.5, x_initial)
if isinstance(result_a1, str):
    print(f"Alpha = 0.5: {result_a1}")
else:
    x_a1, optimal_value_a1, iterations_a1 = result_a1
    print(f"Alpha = 0.5: \nOptimal Decision Variables x* = {x_a1} \nOptimal Value = {optimal_value_a1} \nIterations = {iterations_a1}")

# Run with alpha = 0.9
result_a2 = interior_point_method(C, A, eps, 0.9, x_initial)
if isinstance(result_a2, str):
    print(f"Alpha = 0.9: {result_a2}")
else:
    x_a2, optimal_value_a2, iterations_a2 = result_a2
    print(f"Alpha = 0.9: \nOptimal Decision Variables x* = {x_a2} \nOptimal Value = {optimal_value_a2} \nIterations = {iterations_a2}")