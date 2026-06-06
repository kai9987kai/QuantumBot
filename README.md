# QuantumBot

QuantumBot is an evolutionary multi-agent laboratory where every bot's movement
is produced by a shallow variational quantum circuit. Bots sense resources and
hazards, share short-lived resource memories with nearby peers, manage energy,
and evolve circuit parameters over successive generations.

The default backend is a deterministic two-qubit NumPy statevector simulator.
Qiskit 2.4+ is optional and can execute the same circuit for validation and
experimentation. This project is a simulation and does not claim quantum
advantage.

## Highlights

- Real parameterized circuit with angle encoding, data re-uploading, and CNOT
  entanglement
- Evolutionary learning with elitism, crossover, mutation, and adaptive
  mutation pressure after stalled generations
- Local sensing, hazards, energy, resource collection, and peer communication
- Reproducible seeded experiments and a headless mode for benchmarks
- Responsive Tkinter UI with no cross-thread widget access
- Optional Qiskit backend using current `Statevector` APIs

## Install

```powershell
python -m pip install -e .
```

For Qiskit validation:

```powershell
python -m pip install -e ".[quantum]"
```

For development:

```powershell
python -m pip install -e ".[dev]"
```

## Run

Launch the visual simulation:

```powershell
python main.py
```

Run a reproducible headless experiment:

```powershell
python main.py --headless --generations 10 --steps 450 --seed 7
```

In the UI, left-click to add a resource and right-click to add a hazard.

## Test

```powershell
python -m pytest
```

## Research basis

The circuit stays deliberately shallow and uses local observables to limit the
trainability problems associated with deep, highly expressive variational
circuits. Repeated feature encoding follows the data re-uploading approach used
to improve compact quantum models. Evolutionary optimization avoids requiring
gradient estimates from sampled quantum executions.

- [Qiskit primitives and statevector simulation](https://docs.quantum.ibm.com/api/qiskit/primitives)
- [Data re-uploading optimization in VQC reinforcement learning](https://arxiv.org/abs/2405.12354)
- [Review of barren plateaus in variational quantum computing](https://arxiv.org/abs/2405.00781)
