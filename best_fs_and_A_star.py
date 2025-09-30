import heapq

def best_fs(graph, start, goal, h):
    open_list = [(h[start], start)]                        # h = heuristic
    visited = []
    result = []
    total_cost = 0
    
    while open_list:
        f = [node for _, node in open_list]
        print(f"Frontier: {f}")                            # f = frontier
        print(f"Explored: {visited}")
        print("------------------------")
        
        _, current = heapq.heappop(open_list)
        if current == goal:
            result.append(current)
            total_cost = sum(costs[(result[i], result[i + 1])] for i in range(len(result) - 1))
            return result, total_cost
        
        if current in visited:
            continue
        
        visited.append(current)
        result.append(current)
        
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                heapq.heappush(open_list, (h[neighbor], neighbor))
    
    return result, total_cost

def a_star(graph, start, goal, h, cost):
    open_list = [(0 + h[start], start)]  
    g = {start: 0}                                       # g = g(n)
    came_from = {}
    frontier = [] 
    explored = [] 
    result = []
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        # Print frontier and explored lists
        frontier = [node for _, node in open_list]  # Update frontier list
        explored.append(current)
        
        
        print(f"Frontier: {frontier}")
        print(f"Explored: {explored}")
        print(f"------------------------")
        
        if current == goal:
            while current in came_from:
                result.append(current)
                current = came_from[current]
            result.append(start)
            result.reverse()
            total_cost = g.get(goal, float('inf'))
            return result, total_cost
                                             # tg = tentative g(n)
        for neighbor in graph[current]:
            tg = g[current] + cost[(current, neighbor)]
            if neighbor not in g or tg < g[neighbor]:
                g[neighbor] = tg
                f_cost = tg + h[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))
                came_from[neighbor] = current
    
    return result, float('inf')

def read_graph():
    graph = {}
    costs = {}
    num_nodes = int(input("Enter the number of nodes: "))
    num_edges = int(input("Enter the number of edges: "))
    
    for _ in range(num_edges):
        u, v, w = input("Enter an edge with cost (e.g., 'A B 1'): ").split()
        w = int(w)
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
        costs[(u, v)] = w
        costs[(v, u)] = w
    
    return graph, costs

def read_heuristics(num_nodes):
    heuristics = {}
    for _ in range(num_nodes):
        node, h = input("Enter node and heuristic value (e.g., 'A 10'): ").split()
        heuristics[node] = int(h)
    return heuristics


graph, costs = read_graph()
num_nodes = len(graph)
heuristic = read_heuristics(num_nodes)

start_node = input("Enter the starting node: ")
goal_node = input("Enter the goal node: ")

print('****************  BEST FIRST SEARCH  ****************')
bfs_result, bfs_cost = best_fs(graph, start_node, goal_node, heuristic)
print(f"Total cost to reach the goal: {bfs_cost}")
print(f"Best-First Search Path: {bfs_result}")

print('****************  A STAR SEARCH  ****************')
a_star_result, a_star_cost = a_star(graph, start_node, goal_node, heuristic, costs)
print(f"Total cost to reach the goal: {a_star_cost}")
print(f"A* Search Path: {a_star_result}")
