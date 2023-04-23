import random
import matplotlib.pyplot as plt

random.seed(42)

def burst_noise_simulation(num_bits, p, P, h, initial_energy):
    state = 'G'  # Start in the Good state
    transmitted_bits = []
    received_bits = []

    errors_in_good_state = 0
    errors_in_bad_state = 0
    remaining_energy = initial_energy
    energy_cost_per_transmission = 0.00005
    energy_cost_on_retransmission = 0.01

    energy_levels = [remaining_energy]

    for _ in range(num_bits):
        if remaining_energy <= 0:
            break

        # Generate a random bit
        bit = random.choice([0, 1])
        transmitted_bits.append(bit)

        # Simulate transmission based on the current state
        if state == 'G':
            received_bits.append(bit)
            remaining_energy -= energy_cost_per_transmission
            # Transition to the Bad state with probability P
            if random.random() < P:
                state = 'B'
        else:  # state == 'B'
            # Transmit the bit correctly with probability h
            if random.random() < h:
                received_bits.append(bit)
                remaining_energy -= energy_cost_per_transmission
            else:
                received_bits.append(1 - bit)
                errors_in_bad_state += 1
                remaining_energy -= energy_cost_on_retransmission
            # Transition to the Good state with probability p
            if random.random() < p:
                state = 'G'
        if state == 'G' and bit != received_bits[-1]:
            errors_in_good_state += 1

        energy_levels.append(remaining_energy)

    return transmitted_bits, received_bits, errors_in_good_state, errors_in_bad_state, remaining_energy, energy_levels

# Simulation parameters
num_bits = 1000
P = 0.03  # B -> G transition probability
p = 0.25  # G -> B transition probability
h = 0.01  # Probability of correct transmission in state B
initial_energy = 5

transmitted_bits, received_bits, errors_in_good_state, errors_in_bad_state, remaining_energy, energy_levels = burst_noise_simulation(num_bits, p, P, h, initial_energy)

error_count = sum([t_bit != r_bit for t_bit, r_bit in zip(transmitted_bits, received_bits)])

print("Number of errors:", error_count)
print("Error rate:", error_count / len(transmitted_bits))
print("Errors in good state:", errors_in_good_state)
print("Errors in bad state:", errors_in_bad_state)
print("Remaining energy:", remaining_energy)

# Plot the energy levels
plt.plot(energy_levels)
plt.xlabel('Transmission Step')
plt.ylabel('Remaining Energy')
plt.title('Energy Reduction over Time')
plt.grid()
plt.show()
