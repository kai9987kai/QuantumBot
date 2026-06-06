"""Deterministic evolutionary multi-agent simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

import numpy as np
from numpy.typing import NDArray

from .quantum import PARAMETER_COUNT, VariationalQuantumPolicy

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    width: int = 900
    height: int = 640
    population_size: int = 24
    resource_count: int = 32
    hazard_count: int = 8
    generation_steps: int = 450
    max_energy: float = 100.0
    movement_speed: float = 4.0
    metabolism: float = 0.12
    movement_cost: float = 0.04
    resource_energy: float = 28.0
    hazard_damage: float = 24.0
    sensor_range: float = 190.0
    communication_range: float = 95.0
    memory_steps: int = 90
    bot_radius: float = 5.0
    resource_radius: float = 4.0
    hazard_radius: float = 12.0
    elite_fraction: float = 0.2
    mutation_rate: float = 0.18
    mutation_scale: float = 0.16

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        minimum_dimension = 2 * max(
            self.bot_radius, self.resource_radius, self.hazard_radius
        )
        if self.width <= minimum_dimension or self.height <= minimum_dimension:
            raise ValueError("world dimensions are too small for configured objects")
        if self.generation_steps < 1:
            raise ValueError("generation_steps must be positive")
        if self.resource_count < 0 or self.hazard_count < 0:
            raise ValueError("object counts cannot be negative")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")


@dataclass(slots=True)
class Point:
    x: float
    y: float


@dataclass(slots=True)
class Agent:
    identifier: int
    x: float
    y: float
    genome: FloatArray
    energy: float
    score: float = 0.0
    resources_collected: int = 0
    alive: bool = True
    memory: Point | None = None
    memory_ttl: int = 0
    last_action: FloatArray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    policy: VariationalQuantumPolicy = field(init=False)

    def __post_init__(self) -> None:
        self.policy = VariationalQuantumPolicy(self.genome)


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    generation: int
    best_score: float
    mean_score: float
    resources_collected: int
    mutation_scale: float


class Simulation:
    def __init__(
        self, config: SimulationConfig | None = None, *, seed: int = 7
    ) -> None:
        self.config = config or SimulationConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.generation = 1
        self.step_count = 0
        self.best_score_ever = float("-inf")
        self.stagnant_generations = 0
        self.history: list[GenerationSummary] = []
        self.resources: list[Point] = []
        self.hazards: list[Point] = []
        self.agents: list[Agent] = []
        self._next_identifier = 0
        self._reset_world()

    @property
    def alive_count(self) -> int:
        return sum(agent.alive for agent in self.agents)

    @property
    def best_agent(self) -> Agent:
        return max(self.agents, key=lambda agent: agent.score)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.generation = 1
        self.step_count = 0
        self.best_score_ever = float("-inf")
        self.stagnant_generations = 0
        self.history.clear()
        self._next_identifier = 0
        self._reset_world()

    def add_resource(self, x: float, y: float) -> None:
        self.resources.append(self._clamped_point(x, y))

    def add_hazard(self, x: float, y: float) -> None:
        self.hazards.append(self._clamped_point(x, y))

    def step(self) -> GenerationSummary | None:
        self._update_shared_memory()
        for agent in self.agents:
            if agent.alive:
                self._step_agent(agent)

        self.step_count += 1
        if self.step_count >= self.config.generation_steps or self.alive_count == 0:
            return self.evolve()
        return None

    def run_generation(self) -> GenerationSummary:
        starting_generation = self.generation
        while self.generation == starting_generation:
            summary = self.step()
            if summary is not None:
                return summary
        raise RuntimeError("Generation ended without a summary")

    def evolve(self) -> GenerationSummary:
        scores = np.array([agent.score for agent in self.agents], dtype=np.float64)
        generation_best = float(np.max(scores))
        summary = GenerationSummary(
            generation=self.generation,
            best_score=generation_best,
            mean_score=float(np.mean(scores)),
            resources_collected=sum(agent.resources_collected for agent in self.agents),
            mutation_scale=self._adaptive_mutation_scale(generation_best),
        )
        self.history.append(summary)

        ranked = sorted(self.agents, key=lambda agent: agent.score, reverse=True)
        elite_count = max(
            1, round(self.config.population_size * self.config.elite_fraction)
        )
        next_genomes = [agent.genome.copy() for agent in ranked[:elite_count]]
        while len(next_genomes) < self.config.population_size:
            parent_a = self._tournament_select(ranked)
            parent_b = self._tournament_select(ranked)
            child = self._crossover(parent_a.genome, parent_b.genome)
            self._mutate(child, summary.mutation_scale)
            next_genomes.append(child)

        self.generation += 1
        self.step_count = 0
        self.resources = [
            self._random_point() for _ in range(self.config.resource_count)
        ]
        self.hazards = [self._random_point() for _ in range(self.config.hazard_count)]
        self.agents = [self._new_agent(genome) for genome in next_genomes]
        return summary

    def _reset_world(self) -> None:
        self.resources = [
            self._random_point() for _ in range(self.config.resource_count)
        ]
        self.hazards = [self._random_point() for _ in range(self.config.hazard_count)]
        self.agents = [
            self._new_agent(VariationalQuantumPolicy.random(self.rng).parameters.copy())
            for _ in range(self.config.population_size)
        ]

    def _new_agent(self, genome: FloatArray) -> Agent:
        point = self._random_point()
        agent = Agent(
            identifier=self._next_identifier,
            x=point.x,
            y=point.y,
            genome=np.asarray(genome, dtype=np.float64).reshape(PARAMETER_COUNT),
            energy=self.config.max_energy,
        )
        self._next_identifier += 1
        return agent

    def _step_agent(self, agent: Agent) -> None:
        observations = self._observations(agent)
        action = agent.policy.act(observations)
        norm = float(np.linalg.norm(action))
        if norm > 1.0:
            action /= norm
        dx, dy = action * self.config.movement_speed
        old_x, old_y = agent.x, agent.y
        agent.x = float(np.clip(agent.x + dx, 0.0, self.config.width))
        agent.y = float(np.clip(agent.y + dy, 0.0, self.config.height))
        agent.last_action = action

        distance_moved = hypot(agent.x - old_x, agent.y - old_y)
        agent.energy -= (
            self.config.metabolism + distance_moved * self.config.movement_cost
        )
        agent.score += 0.01

        self._collect_resource(agent)
        self._apply_hazard_damage(agent)
        if agent.energy <= 0.0:
            agent.energy = 0.0
            agent.alive = False

    def _observations(self, agent: Agent) -> FloatArray:
        resource_dx, resource_dy = self._direction_features(agent, agent.memory)
        nearest_hazard = self._nearest_point(
            agent.x, agent.y, self.hazards, self.config.sensor_range
        )
        hazard_dx, hazard_dy = self._direction_features(agent, nearest_hazard)
        energy = 2.0 * agent.energy / self.config.max_energy - 1.0
        peers = sum(
            other.alive
            and other is not agent
            and hypot(other.x - agent.x, other.y - agent.y)
            <= self.config.communication_range
            for other in self.agents
        )
        peer_density = 2.0 * min(peers / 5.0, 1.0) - 1.0
        return np.array(
            [
                resource_dx,
                resource_dy,
                hazard_dx,
                hazard_dy,
                energy,
                peer_density,
            ],
            dtype=np.float64,
        )

    def _update_shared_memory(self) -> None:
        for agent in self.agents:
            if not agent.alive:
                continue
            visible = self._nearest_point(
                agent.x, agent.y, self.resources, self.config.sensor_range
            )
            if visible is not None:
                agent.memory = Point(visible.x, visible.y)
                agent.memory_ttl = self.config.memory_steps
            elif agent.memory_ttl > 0:
                agent.memory_ttl -= 1
            else:
                agent.memory = None

        for index, agent in enumerate(self.agents):
            if not agent.alive:
                continue
            for other in self.agents[index + 1 :]:
                if not other.alive:
                    continue
                if (
                    hypot(other.x - agent.x, other.y - agent.y)
                    > self.config.communication_range
                ):
                    continue
                if other.memory_ttl > agent.memory_ttl and other.memory is not None:
                    agent.memory = Point(other.memory.x, other.memory.y)
                    agent.memory_ttl = other.memory_ttl
                elif agent.memory_ttl > other.memory_ttl and agent.memory is not None:
                    other.memory = Point(agent.memory.x, agent.memory.y)
                    other.memory_ttl = agent.memory_ttl

    def _collect_resource(self, agent: Agent) -> None:
        collision_distance = self.config.bot_radius + self.config.resource_radius
        for index, resource in enumerate(self.resources):
            if hypot(resource.x - agent.x, resource.y - agent.y) <= collision_distance:
                agent.resources_collected += 1
                agent.score += 10.0
                agent.energy = min(
                    self.config.max_energy,
                    agent.energy + self.config.resource_energy,
                )
                agent.memory = None
                agent.memory_ttl = 0
                self.resources[index] = self._random_point()
                return

    def _apply_hazard_damage(self, agent: Agent) -> None:
        collision_distance = self.config.bot_radius + self.config.hazard_radius
        for hazard in self.hazards:
            if hypot(hazard.x - agent.x, hazard.y - agent.y) <= collision_distance:
                agent.energy -= self.config.hazard_damage
                agent.score -= 3.0
                away_x = agent.x - hazard.x
                away_y = agent.y - hazard.y
                length = hypot(away_x, away_y) or 1.0
                agent.x = float(
                    np.clip(
                        agent.x + away_x / length * self.config.hazard_radius,
                        0.0,
                        self.config.width,
                    )
                )
                agent.y = float(
                    np.clip(
                        agent.y + away_y / length * self.config.hazard_radius,
                        0.0,
                        self.config.height,
                    )
                )
                return

    def _direction_features(
        self, agent: Agent, point: Point | None
    ) -> tuple[float, float]:
        if point is None:
            return 0.0, 0.0
        scale = self.config.sensor_range
        return (
            float(np.clip((point.x - agent.x) / scale, -1.0, 1.0)),
            float(np.clip((point.y - agent.y) / scale, -1.0, 1.0)),
        )

    @staticmethod
    def _nearest_point(
        x: float, y: float, points: list[Point], max_distance: float
    ) -> Point | None:
        nearest = None
        nearest_distance = max_distance
        for point in points:
            distance = hypot(point.x - x, point.y - y)
            if distance <= nearest_distance:
                nearest = point
                nearest_distance = distance
        return nearest

    def _adaptive_mutation_scale(self, generation_best: float) -> float:
        if generation_best > self.best_score_ever + 1e-9:
            self.best_score_ever = generation_best
            self.stagnant_generations = 0
        else:
            self.stagnant_generations += 1
        multiplier = min(2.5, 1.0 + 0.18 * self.stagnant_generations)
        return self.config.mutation_scale * multiplier

    def _tournament_select(self, ranked: list[Agent], size: int = 4) -> Agent:
        indices = self.rng.integers(0, len(ranked), size=size)
        contestants = [ranked[int(index)] for index in indices]
        return max(contestants, key=lambda agent: agent.score)

    def _crossover(self, parent_a: FloatArray, parent_b: FloatArray) -> FloatArray:
        blend = self.rng.uniform(0.0, 1.0, size=PARAMETER_COUNT)
        return blend * parent_a + (1.0 - blend) * parent_b

    def _mutate(self, genome: FloatArray, scale: float) -> None:
        mask = self.rng.random(PARAMETER_COUNT) < self.config.mutation_rate
        genome[mask] += self.rng.normal(0.0, scale, size=int(np.sum(mask)))
        np.clip(genome, -np.pi, np.pi, out=genome)

    def _random_point(self) -> Point:
        margin = max(
            self.config.bot_radius,
            self.config.resource_radius,
            self.config.hazard_radius,
        )
        return Point(
            float(self.rng.uniform(margin, self.config.width - margin)),
            float(self.rng.uniform(margin, self.config.height - margin)),
        )

    def _clamped_point(self, x: float, y: float) -> Point:
        return Point(
            float(np.clip(x, 0.0, self.config.width)),
            float(np.clip(y, 0.0, self.config.height)),
        )
