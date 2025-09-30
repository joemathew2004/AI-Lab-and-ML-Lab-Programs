def is_valid(assignment, variable, value, constraints):
    for (var1, var2) in constraints:
        if var1 == variable and var2 in assignment and assignment[var2] == value:
            return False
        if var2 == variable and var1 in assignment and assignment[var1] == value:
            return False
    return True

def get_available_colors(assignment, variable, domains, constraints):
    available = []
    for value in domains:
        if is_valid(assignment, variable, value, constraints):
            available.append(value)
    return available

def backtrack(variables, domains, assignment, constraints):
    if len(assignment) == len(variables):
        return assignment
    
    unassigned_vars = [v for v in variables if v not in assignment]
    variable = unassigned_vars[0]
    
    available_colors = get_available_colors(assignment, variable, domains, constraints)
    print(f"{variable} can be assigned to: {', '.join(available_colors)}")
    
    for value in available_colors:
        assignment[variable] = value
        print(f"Assigning {value} to variable {variable}: {assignment}")
        
        result = backtrack(variables, domains, assignment, constraints)
        if result is not None:
            return result
        
        print(f"BUT THIS IS AGAINST THE CONSTRAINT: {value} cannot be assigned to variable {variable}")
        print("SO BACKTRACKING")
        del assignment[variable]
    
    print(f"All values for variable {variable} have been tried. Backtracking...")
    return None

def main():
    variables = input("Enter the set of variables (comma-separated): ").split(',')
    variables = [var.strip() for var in variables]

    domains = input("Enter the set of domains (comma-separated): ").split(',')
    domains = [domain.strip() for domain in domains]

    constraints_input = input("Enter the set of constraints (e.g., 1 not equal to 2, 1 not equal to 3): ")
    constraints = []
    for constraint in constraints_input.split(','):
        var1, var2 = constraint.strip().split(' not equal to ')
        constraints.append((var1.strip(), var2.strip()))

    assignment = {}
    result = backtrack(variables, domains, assignment, constraints)
    
    if result:
        print("Solution found:", result)
    else:
        print("No solution exists.")

if __name__ == "__main__":
    main()
