def minimax(node, depth, maximizingPlayer, tree, terminal_values):
    if depth == 0 or node in terminal_values:
        return terminal_values[node], [node]

    if maximizingPlayer:
        maxEval = float('-inf')
        best_path = []
        for child in tree[node]:
            eval, path = minimax(child, depth - 1, False, tree, terminal_values)
            if eval > maxEval:
                maxEval = eval
                best_path = [node] + path
        return maxEval, best_path
    else:
        minEval = float('inf')
        best_path = []
        for child in tree[node]:
            eval, path = minimax(child, depth - 1, True, tree, terminal_values)
            if eval < minEval:
                minEval = eval
                best_path = [node] + path
        return minEval, best_path

def alpha_beta(node, depth, alpha, beta, maximizingPlayer, tree, terminal_values):
    if depth == 0 or node in terminal_values:
        return terminal_values[node], [node]

    if maximizingPlayer:
        maxEval = float('-inf')
        best_path = []
        for child in tree[node]:
            eval, path = alpha_beta(child, depth - 1, alpha, beta, False, tree, terminal_values)
            if eval > maxEval:
                maxEval = eval
                best_path = [node] + path
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return maxEval, best_path
    else:
        minEval = float('inf')
        best_path = []
        for child in tree[node]:
            eval, path = alpha_beta(child, depth - 1, alpha, beta, True, tree, terminal_values)
            if eval < minEval:
                minEval = eval
                best_path = [node] + path
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return minEval, best_path

def build_tree_and_values(depth):
    tree = {}
    terminal_values = {}

    for d in range(depth):
        print(f"Enter the nodes at depth {d}:")
        nodes = input().split()
        tree.update({node: [] for node in nodes})
        if d > 0:
            for parent in list(tree.keys()):
                if len(tree[parent]) < 2:  # binary tree structure
                    tree[parent].extend(nodes[:2])
                    nodes = nodes[2:]

    print("Enter terminal node values:")
    for node in tree:
        if not tree[node]:  # terminal node
            value = int(input(f"Value of node {node}: "))
            terminal_values[node] = value

    return tree, terminal_values

def main():
    depth = int(input("Enter the depth of the tree: "))
    tree, terminal_values = build_tree_and_values(depth)

    root = list(tree.keys())[0]  # root
    
    # Perform Minimax
    optimal_value_minimax, optimal_path_minimax = minimax(root, depth, True, tree, terminal_values)
    print("\n--- Minimax Results ---")
    print(f"Optimal value: {optimal_value_minimax}")
    print(f"Optimal path: {' -> '.join(optimal_path_minimax)}")

    # Perform Alpha-Beta Pruning
    alpha = float('-inf')
    beta = float('inf')
    optimal_value_alpha_beta, optimal_path_alpha_beta = alpha_beta(root, depth, alpha, beta, True, tree, terminal_values)
    print("\n--- Alpha-Beta Pruning Results ---")
    print(f"Optimal value: {optimal_value_alpha_beta}")
    print(f"Optimal path: {' -> '.join(optimal_path_alpha_beta)}")

if __name__ == "__main__":
    main()