import threading

import logging
import random
import time

import math
import tkinter as tk

import torch
from torch import nn
from torch.nn import functional as F
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set up the window
root = tk.Tk()
canvas = tk.Canvas(root, width=800, height=800)
canvas.pack()

# Set up the genetic algorithm parameters
population_size = 10
mutation_rate = 0.1

# Set up the neural network parameters
num_inputs = 4  # Updated to include sight and hearing inputs
num_hidden = 20
num_outputs = 2

# Set up the quantum circuit parameters
n_qubits = 2
theta_parameter = random.random()

# Create a quantum circuit with 2 qubits
qc = QuantumCircuit(n_qubits)

# Apply Hadamard gate to the first qubit
qc.h(0)

# Apply a CNOT gate
qc.cx(0, 1)

# Measure the qubits
qc.measure_all()


# Define the quantum layer
class QuantumLayer(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float))

    def forward(self):
        # Execute the quantum circuit on a simulator
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1, parameter_binds=[{Parameter('theta'): self.theta.item()}])
        result = job.result()
        counts = result.get_counts(qc)

        # Convert the counts to a tensor
        counts_tensor = torch.tensor(
            [counts.get('00', 0), counts.get('01', 0), counts.get('10', 0), counts.get('11', 0)], dtype=torch.float)
        return counts_tensor


# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.quantum_layer = QuantumLayer(theta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)  # Updated to num_outputs

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.quantum_layer(x)
        x = F.relu(self.fc2(x))  # Updated to use the second linear layer
        return x


# Define the bot
class Bot:
    def __init__(self, x, y, canvi, theta):
        self.x = x
        self.y = y
        self.canvas = canvi
        self.rect = canvi.create_rectangle(x, y, x + 10, y + 10, fill='blue')
        self.model = NeuralNetwork(theta)
        self.resources = []
        self.threats = []

    def move(self):
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)

        pos = self.canvas.coords(self.rect)

        # Reverse direction
        if pos[0] + dx < 0 or pos[0] + dx > 790:
            dx = -dx  # Reverse direction

        # Reverse direction
        if pos[1] + dy < 0 or pos[1] + dy > 790:
            dy = -dy  # Reverse direction

        self.canvas.delete(self.rect)

        self.x += dx
        self.y += dy
        self.rect = self.canvas.create_rectangle(self.x, self.y, self.x + 10, self.y + 10, fill='blue')

    def share_resources(self, other_bots):
        for other in other_bots:
            if other != self and self.distance(other) < 100:
                other.resources.extend(self.resources)
                other.resources = list(set(other.resources))

    def share_threats(self, other_bots):
        for other in other_bots:
            if other != self and self.distance(other) < 100:
                other.threats.extend(self.threats)
                other.threats = list(set(other.threats))

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# Create multiple bots with random initial parameters
bots = [Bot(random.randint(0, 790), random.randint(0, 790), canvas, random.random()) for _ in range(10)]

# Create plants as resources
plants = [Bot(random.randint(0, 790), random.randint(0, 790), canvas, random.random()) for _ in range(5)]
for plant in plants:
    plant.rect = canvas.create_rectangle(plant.x, plant.y, plant.x + 5, plant.y + 5, fill='green')

# Create hostile bots as threats
threats = [Bot(random.randint(0, 790), random.randint(0, 790), canvas, random.random()) for _ in range(3)]
for threat in threats:
    threat.rect = canvas.create_rectangle(threat.x, threat.y, threat.x + 10, threat.y + 10, fill='red')


# Move the bots indefinitely
def move_bots():
    while True:
        for bot in bots:
            bot.move()
        time.sleep(0.1)


# Start the bot movement thread
move_thread = threading.Thread(target=move_bots)
move_thread.start()


# Share resources and threats among bots
def share_resources_and_threats():
    while True:
        for bot in bots:
            bot.share_resources(bots)
            bot.share_resources(plants)
            bot.share_threats(bots)
            bot.share_threats(threats)
        time.sleep(1)


# Start the sharing thread
share_thread = threading.Thread(target=share_resources_and_threats)
share_thread.start()

# Start the main loop
root.mainloop()
