import numpy as np

environment_rows = 5
environment_columns = 5

STARTING_LOCATION = [4, 4]

q_values = np.zeros((environment_rows, environment_columns, 8))

actions = ['up', 'right', 'down', 'left', 'left up diagonal', 'right up diagonal', 'left down diagonal',  'right down diagonal']

rewards = np.full((environment_rows, environment_columns), -100)


aisles = {0: [0, 1], 1: [0, 1, 2, 3], 2: [1, 3, 4], 3: [1, 4], 4: [1, 2, 3, 4]}

for row_index in range(0, 5):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1

rewards[0, 0] = 100

print(rewards)

def is_terminal_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1:
        return False
    return True


def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    return np.random.randint(8)


def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index

    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    elif actions[action_index] == 'left up diagonal' and current_row_index > 0 and current_column_index > 0:
        new_row_index -= 1
        new_column_index -= 1
    elif actions[action_index] == 'right up diagonal' and current_row_index > 0 and current_column_index < environment_columns - 1:
        new_row_index -= 1
        new_column_index += 1
    elif actions[action_index] == 'left down diagonal' and current_row_index < environment_rows - 1 and current_column_index > 0:
        new_row_index += 1
        new_column_index -= 1
    elif actions[action_index] == 'right down diagonal' and current_row_index < environment_rows - 1 and current_column_index < environment_columns - 1:
        new_row_index += 1
        new_column_index += 1

    return new_row_index, new_column_index


def get_shortest_path(start_row_index, start_column_index):
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1)
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index,
                                                                        action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path


epsilon = 0.8
discount_factor = 0.8
learning_rate = 0.8

for episode in range(1000):
    row_index, column_index = STARTING_LOCATION

    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)

        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')

print(get_shortest_path(STARTING_LOCATION[0], STARTING_LOCATION[1]))