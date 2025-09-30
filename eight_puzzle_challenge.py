import heapq
import time
import tkinter as tk

GOAL_STATE = ((1, 2, 3), (4, 5, 6), (7, 8, 0))

def manhattan_distance(state):
    distance = 0
    for r in range(3):
        for c in range(3):
            value = state[r][c]
            if value != 0:
                goal_r, goal_c = divmod(value - 1, 3)
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

def get_neighbors(state):
    def swap(state, r1, c1, r2, c2):
        state = [list(row) for row in state]
        state[r1][c1], state[r2][c2] = state[r2][c2], state[r1][c1]
        return tuple(tuple(row) for row in state)

    neighbors = []
    zero_r, zero_c = [(r, c) for r in range(3) for c in range(3) if state[r][c] == 0][0]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for move in moves:
        new_r, new_c = zero_r + move[0], zero_c + move[1]
        if 0 <= new_r < 3 and 0 <= new_c < 3:
            neighbors.append(swap(state, zero_r, zero_c, new_r, new_c))

    return neighbors

def best_first_search(start_state):
    open_list = [(manhattan_distance(start_state), start_state)]
    heapq.heapify(open_list)
    came_from = {start_state: None}
    cost_so_far = {start_state: 0}

    while open_list:
        _, current_state = heapq.heappop(open_list)
        
        if current_state == GOAL_STATE:
            path = []
            while current_state:
                path.append(current_state)
                current_state = came_from[current_state]
            path.reverse()
            return path

        for neighbor in get_neighbors(current_state):
            new_cost = cost_so_far[current_state] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan_distance(neighbor)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current_state

    return []

def read_start_state():
    state = [
        (2, 1, 3),
        (4, 0, 5),
        (6, 8, 7)
    ]
    return tuple(state)

def update_gui_puzzle(state, labels):
    for r in range(3):
        for c in range(3):
            value = state[r][c]
            labels[r][c].config(text=str(value) if value != 0 else '', bg='YELLOW' if value != 0 else 'white')

def solve_puzzle():
    for state in path:
        if puzzle_solved_var.get() == 1:
            break
        update_gui_puzzle(state, labels)
        root.update()
        time.sleep(0.77)

    if puzzle_solved_var.get() == 0:
        success_label.config(text="The puzzle is solved! \nBetter Luck next time!!", fg='red')

def start_puzzle():
    puzzle_solved_var.set(0)
    solve_puzzle()  # Start solving and animating the puzzle


def stop_puzzle():
    puzzle_solved_var.set(1)
    success_label.config(text="Congratulations!! \nYou have won the Challenge", fg='green')


root = tk.Tk()
root.title("8-Puzzle")

root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

puzzle_frame = tk.Frame(root)
puzzle_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

labels = [[tk.Label(puzzle_frame, font=('Arial', 60), width=4, height=2, borderwidth=2, relief="solid") for _ in range(3)] for _ in range(3)]
for r in range(3):
    for c in range(3):
        labels[r][c].grid(row=r, column=c, padx=20, pady=20)

success_label = tk.Label(root, text="", font=('Arial', 30), fg='red', bg='white')
success_label.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-20, y=-20)

start_button = tk.Button(root, text="Start", font=('Arial', 20), command=start_puzzle)
start_button.place(relx=0.0, rely=1.0, anchor=tk.SW, x=20, y=-20)

stop_button = tk.Button(root, text="Stop", font=('Arial', 20), command=stop_puzzle)
stop_button.place(relx=0.06, rely=1.0, anchor=tk.SW, x=20, y=-20)

puzzle_solved_var = tk.IntVar(value=0)

start_state = read_start_state()
path = best_first_search(start_state)
root.mainloop()
