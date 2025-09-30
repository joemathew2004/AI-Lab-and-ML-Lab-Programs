from collections import deque

def water_jug_problem_bfs():
    capacity1 = 6
    capacity2 = 8
    target = 4
    
    initial_state = (0, 0)
    
    queue = deque([(initial_state, [])])          # Queue holds (state, path) 
    visited = set()                             # To track visited states
    visited.add(initial_state)
    
    while queue:
        (x, y), path = queue.popleft()
        
        if x == target:
            output = []
            output.append("Steps to measure exactly 4 liters in Jug A:")
            for i, (a, b, action) in enumerate(path + [(x, y, "Goal State")]):
                output.append(f"Step {i}: Jug A = {a} liters, Jug B = {b} liters - Action: {action}")
            return "\n".join(output)
        
        next_states = [
            ((capacity1, y), "Fill Jug A"),                    # Fill Jug A
            ((x, capacity2), "Fill Jug B"),                    # Fill Jug B
            ((0, y), "Empty Jug A"),                                # Empty Jug A
            ((x, 0), "Empty Jug B"),                                # Empty Jug B
            ((max(x - (capacity2 - y), 0), min(y + x, capacity2)), "Pour A -> B"),    # Pour A -> B
            ((min(x + y, capacity1), max(y - (capacity1 - x), 0)), "Pour B -> A")     # Pour B -> A
        ]
        
        for (new_a, new_b), action in next_states:
            new_state = (new_a, new_b)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [(x, y, action)]))  # Append new state and updated path

    return "No solution found to measure exactly 4 liters in Jug A."

print(water_jug_problem_bfs())
