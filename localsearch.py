import random

def random_coloring(variables, domains):
    return {region: random.choice(domains) for region in variables}

def count_conflicts(graph, coloring):
    conflicts = 0
    for region, neighbors in graph.items():
        for neighbor in neighbors:
            if coloring[region] == coloring[neighbor]:
                conflicts += 1
    return conflicts

def hill_climbing(graph, variables, domains):
    current_coloring = random_coloring(variables, domains)
    current_conflicts = count_conflicts(graph, current_coloring)

    # Step 1: Random coloring
    print("\nInitial Random Coloring:", current_coloring)
    print("Initial Conflicts:", current_conflicts)

    step = 0

    while current_conflicts > 0:
        improved = False

        for region in variables:
            original_color = current_coloring[region]

            for color in domains:
                if color != original_color:
                    current_coloring[region] = color
                    new_conflicts = count_conflicts(graph, current_coloring)

                    if new_conflicts < current_conflicts:
                        current_conflicts = new_conflicts
                        improved = True

                        print(f"\nStep {step}: Changed color of {region} to {color}")
                        print("Current Coloring:", current_coloring)
                        print("Conflicts:", current_conflicts)
                        break  # Exit inner loop to start on the next improvement

            # Restore the original color if no improvement was made for this region
            if not improved:
                current_coloring[region] = original_color

        # If no improvements can be made, stop the process (local optimum reached)
        if not improved:
            print(f"Local Optimum reached at Step {step}, Conflicts: {current_conflicts}")
            break 

        step += 1

    print("\nFinal Solution:", current_coloring)
    print("Final Conflicts:", current_conflicts)

    return current_coloring

variables = input("Enter the set of variables (comma-separated): ").split(",")
variables = [var.strip() for var in variables]

domains = input("Enter the set of colors (comma-separated): ").split(",")
domains = [color.strip() for color in domains]

graph = {}

for region in variables:
    neighbors = input(f"Enter the adjacent variables for {region} (comma-separated, or leave blank if none): ")
    graph[region] = [neighbor.strip() for neighbor in neighbors.split(",") if neighbor.strip()]

final_coloring = hill_climbing(graph, variables, domains)
