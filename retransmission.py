import numpy as np
import random

class TransmissionEnvironment:
    def __init__(self, num_bits, P):
        self.num_bits = num_bits
        self.P = P
        self.state = 'G'

    def reset(self):
        self.state = 'G'
        return self.state

    def step(self, action):
        # Simulate transmission
        success = self.simulate_transmission(self.state, action)

        # Calculate reward
        if success:
            reward = 1
        elif action == "transmit":
            reward = -0.1
        elif action == "short_wait" and self.state == "M":
            reward = 3
        elif action == "short_wait":
            reward = -1
        elif action == "long_wait" and self.state == "B":
            reward = 5
        else:  # action == "long_wait"
            reward = -2

        # Transition to the next state
        next_state = np.random.choice(['G', 'M', 'B'], p=self.P[self.state][action])

        self.state = next_state
        return self.state, reward

    def simulate_transmission(self, state, action):
        if action == "transmit":
            return True
        return False

# Define the transition probabilities
P = {
    'G': {
        'transmit': [0.8, 0.15, 0.05],
        'short_wait': [0.9, 0.1, 0],
        'long_wait': [0.95, 0.05, 0],
    },
    'M': {
        'transmit': [0.3, 0.5, 0.2],
        'short_wait': [0.4, 0.5, 0.1],
        'long_wait': [0.6, 0.3, 0.1],
    },
    'B': {
        'transmit': [0.1, 0.6, 0.3],
        'short_wait': [0.05, 0.7, 0.25],
        'long_wait': [0.1, 0.8, 0.1],
    }
}

# Parameters
num_bits = 500
num_episodes = 100
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
actions = ["transmit", "short_wait", "long_wait"]

# State encoding
state_map = {'G': 0, 'M': 1, 'B': 2}

# Initialize Q-table
q_table = np.zeros((3, len(actions)))

env = TransmissionEnvironment(num_bits, P)

for episode in range(num_episodes):
    state = env.reset()

    for _ in range(num_bits):
        # Choose action using epsilon-greedy strategy
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            state_idx = state_map[state]
            action = actions[np.argmax(q_table[state_idx, :])]

        # Interact with the environment
        next_state, reward = env.step(action)

        # Update Q-table
        state_idx = state_map[state]
        next_state_idx = state_map[next_state]
        action_idx = actions.index(action)

        q_table[state_idx, action_idx] += learning_rate * (
            reward
            + discount_factor * np.max(q_table[next_state_idx, :])
            - q_table[state_idx, action_idx]
        )

        # Update state
        # Update state
        state = next_state

# Display the learned policy
for state in ['G', 'M', 'B']:
    state_idx = state_map[state]
    print(f"State {state}:")
    best_action_idx = np.argmax(q_table[state_idx, :])
    best_action = actions[best_action_idx]
    print(f"  Best action: {best_action}")
