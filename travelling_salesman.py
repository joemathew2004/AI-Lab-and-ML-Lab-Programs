from itertools import permutations

def total_distance(graph, path):
    total_dist = 0
    for i in range(len(path) - 1):
        total_dist += graph[path[i]][path[i + 1]]
    total_dist += graph[path[-1]][path[0]]  # Return to starting city
    return total_dist

def travelling_salesman(graph, start, city_names):
    cities = list(range(len(graph)))
    cities.remove(start)

    min_path = None
    min_dist = float('inf')
    all_paths = []

    for perm in permutations(cities):
        current_path = [start] + list(perm) + [start]  # Add starting city to end
        current_dist = total_distance(graph, current_path)

        # Store the current path and its distance
        all_paths.append((current_path, current_dist))

        if current_dist < min_dist:
            min_dist = current_dist
            min_path = current_path

    min_path_names = [city_names[i] for i in min_path]
    return min_path_names, min_dist, all_paths

def main():
    n = int(input("Enter the number of cities: "))
    city_names = []

    for i in range(n):
        city_name = input(f"Enter the name of city {i + 1}: ")
        city_names.append(city_name)

    graph = []
    print("\nDISTANCE MATRIX")
    print("Enter distances from each city to other cities:")

    for i in range(n):
        row = list(map(int, input(f"City ({city_names[i]}) to other cities: ").split()))
        graph.append(row)

    start_city = input("\nEnter the starting city: ")
    start = city_names.index(start_city)

    min_path, min_dist, all_paths = travelling_salesman(graph, start, city_names)

    print("\nAll possible paths and their distances:")
    for path, dist in all_paths:
        path_names = [city_names[i] for i in path]
        print(f"Path: {' -> '.join(path_names)} | Distance: {dist}")

    print("\nMinimum distance:", min_dist)
    print("Optimal path:", " -> ".join(min_path))

if __name__ == "__main__":
    main()
