import numpy as np

from quantumbot.simulation import Point, Simulation, SimulationConfig


def small_config(**overrides: object) -> SimulationConfig:
    values = {
        "width": 120,
        "height": 90,
        "population_size": 6,
        "resource_count": 5,
        "hazard_count": 2,
        "generation_steps": 8,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def test_seed_makes_simulation_reproducible() -> None:
    first = Simulation(small_config(), seed=19)
    second = Simulation(small_config(), seed=19)

    for _ in range(5):
        first.step()
        second.step()

    first_state = [(agent.x, agent.y, agent.score) for agent in first.agents]
    second_state = [(agent.x, agent.y, agent.score) for agent in second.agents]
    np.testing.assert_allclose(first_state, second_state)


def test_generation_evolves_and_preserves_population() -> None:
    simulation = Simulation(small_config(), seed=3)
    original_genomes = [agent.genome.copy() for agent in simulation.agents]

    summary = simulation.run_generation()

    assert summary.generation == 1
    assert simulation.generation == 2
    assert simulation.step_count == 0
    assert len(simulation.agents) == simulation.config.population_size
    assert len(simulation.history) == 1
    assert all(agent.alive for agent in simulation.agents)
    assert any(
        not np.array_equal(agent.genome, original_genomes[index])
        for index, agent in enumerate(simulation.agents)
    )


def test_agents_remain_inside_world() -> None:
    simulation = Simulation(small_config(generation_steps=30), seed=11)

    for _ in range(20):
        simulation.step()

    assert all(
        0.0 <= agent.x <= simulation.config.width
        and 0.0 <= agent.y <= simulation.config.height
        for agent in simulation.agents
    )


def test_nearby_agents_share_resource_memory() -> None:
    config = small_config(
        resource_count=0,
        hazard_count=0,
        sensor_range=20.0,
        communication_range=15.0,
    )
    simulation = Simulation(config, seed=5)
    first, second = simulation.agents[:2]
    first.x, first.y = 30.0, 30.0
    second.x, second.y = 40.0, 30.0
    first.memory = Point(45.0, 35.0)
    first.memory_ttl = 25
    second.memory = None
    second.memory_ttl = 0

    simulation._update_shared_memory()

    assert second.memory is not None
    assert (second.memory.x, second.memory.y) == (45.0, 35.0)
    assert second.memory_ttl == 24
