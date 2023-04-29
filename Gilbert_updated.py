import numpy as np
import random

class TransmissionEnvironment:
    def __init__(self, num_bits, P_GB, P_BG, p_G, p_B):
        self.num_bits = num_bits
        self.P_GB = P_GB
        self.P_BG = P_BG
        self.p_G = p_G
        self.p_B = p_B
        self.state = ('G', 0)

    def reset(self):
        self.state = ('G', 0)
        return self.state

    def step(self, current_timeout, next_timeout):
        channel_state, attempts = self.state

        # Simulate transmission
        success = self.simulate_transmission(channel_state)

        # Calculate reward
        if success:
            reward = 1
        else:
            reward = -0.01  # Small negative reward for sending a packet

        # Additional negative reward for retransmission attempts
        if not success:
            reward -= 0.5 * attempts

        # Transition to the next channel state
        if channel_state == 'G':
            next_channel_state = 'B' if random.random() < self.P_GB else 'G'
        else:
            next_channel_state = 'G' if random.random() < self.P_BG else 'B'

        next_attempts = attempts + 1 if not success else 0
        self.state = (next_channel_state, next_attempts)

        return self.state, reward

    def simulate_transmission(self, channel_state):
        if channel_state == 'G':
            success_prob = self.p_G
        else:  # channel_state == 'B'
            success_prob = self.p_B

        return random.random() < success_prob

# Parameters
num_bits = 5000
P_GB = 0.25  # G -> B transition probability
P_BG = 0.03  # B -> G transition probability
p_G = 0.99   # Probability of correct transmission in state G
p_B = 0.1   # Probability of correct transmission in state B
num_episodes = 1000
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
timeout_values = [1, 2, 4, 8, 16]

# State encoding
state_map = {'G': 0, 'B': 1}
timeout_map = {val: idx for idx, val in enumerate(timeout_values)}

# Initialize Q-table
q_table = np.zeros((2, len(timeout_values), len(timeout_values)))

env = TransmissionEnvironment(num_bits, P_GB, P_BG, p_G, p_B)

for episode in range(num_episodes):
    state = env.reset()
    current_timeout = random.choice(timeout_values)

    for _ in range(num_bits):
        # Choose action using epsilon-greedy strategy
        if random.random() < epsilon:
            next_timeout = random.choice(timeout_values)
        else:
            state_idx = state_map[state[0]]
            current_timeout_idx = timeout_map[current_timeout]
            next_timeout = timeout_values[np.argmax(q_table[state_idx, current_timeout_idx, :])]

        # Interact with the environment
        next_state, reward = env.step(current_timeout, next_timeout)

        # Update Q-table
        state_idx = state_map[state[0]]
        next_state_idx = state_map[next_state[0]]
        current_timeout_idx = timeout_map[current_timeout]
        next_timeout_idx = timeout_map[next_timeout]

        q_table[state_idx, current_timeout_idx, next_timeout_idx] += learning_rate * (
                reward
                + discount_factor * np.max(q_table[next_state_idx, next_timeout_idx, :])
                - q_table[state_idx, current_timeout_idx, next_timeout_idx]
        )

        # Update state and timeout
        state = next_state
        current_timeout = next_timeout

    # Display the learned policy
for channel_state in ['G', 'B']:
    state_idx = state_map[channel_state]
    print(f"Channel state {channel_state}:")
    for current_timeout_idx, current_timeout in enumerate(timeout_values):
        best_next_timeout = timeout_values[np.argmax(q_table[state_idx, current_timeout_idx, :])]
        print(f"  Current timeout {current_timeout}: Best next timeout {best_next_timeout}")


