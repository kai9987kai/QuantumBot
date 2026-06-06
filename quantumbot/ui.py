"""Tkinter visualization for QuantumBot."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .simulation import Simulation


class QuantumBotApp:
    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.root = tk.Tk()
        self.root.title("QuantumBot - Evolutionary Quantum Policy Lab")
        self.root.minsize(760, 560)
        self.running = True
        self.speed = tk.IntVar(value=1)
        self.status = tk.StringVar()

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)
        self.toggle_button = ttk.Button(
            toolbar, text="Pause", command=self._toggle_running
        )
        self.toggle_button.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Reset", command=self._reset).pack(
            side=tk.LEFT, padx=(6, 12)
        )
        ttk.Label(toolbar, text="Steps/frame").pack(side=tk.LEFT)
        ttk.Scale(
            toolbar,
            from_=1,
            to=12,
            variable=self.speed,
            orient=tk.HORIZONTAL,
            length=140,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(toolbar, textvariable=self.status).pack(side=tk.RIGHT)

        config = simulation.config
        self.canvas = tk.Canvas(
            self.root,
            width=config.width,
            height=config.height,
            background="#10151d",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._add_resource)
        self.canvas.bind("<Button-3>", self._add_hazard)

        footer = ttk.Label(
            self.root,
            text="Left click: add resource    Right click: add hazard",
            padding=(8, 4),
        )
        footer.pack(fill=tk.X)

    def run(self) -> None:
        self._frame()
        self.root.mainloop()

    def _frame(self) -> None:
        if self.running:
            for _ in range(max(1, int(self.speed.get()))):
                self.simulation.step()
        self._draw()
        self.root.after(33, self._frame)

    def _draw(self) -> None:
        self.canvas.delete("all")
        config = self.simulation.config
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        scale_x = canvas_width / config.width
        scale_y = canvas_height / config.height

        def coordinates(x: float, y: float, radius: float) -> tuple[float, ...]:
            return (
                (x - radius) * scale_x,
                (y - radius) * scale_y,
                (x + radius) * scale_x,
                (y + radius) * scale_y,
            )

        for resource in self.simulation.resources:
            self.canvas.create_oval(
                *coordinates(resource.x, resource.y, config.resource_radius),
                fill="#4ade80",
                outline="",
            )
        for hazard in self.simulation.hazards:
            self.canvas.create_oval(
                *coordinates(hazard.x, hazard.y, config.hazard_radius),
                fill="#ef4444",
                outline="#fca5a5",
            )

        best = self.simulation.best_agent
        for agent in self.simulation.agents:
            color = "#64748b"
            if agent.alive:
                color = "#fbbf24" if agent is best else "#38bdf8"
            self.canvas.create_oval(
                *coordinates(agent.x, agent.y, config.bot_radius),
                fill=color,
                outline="",
            )
            if agent.alive:
                energy_width = 14.0 * agent.energy / config.max_energy
                left = agent.x * scale_x - 7.0
                top = (agent.y - config.bot_radius - 5.0) * scale_y
                self.canvas.create_rectangle(
                    left,
                    top,
                    left + energy_width,
                    top + 2,
                    fill="#a7f3d0",
                    outline="",
                )

        self.status.set(
            f"Generation {self.simulation.generation}  "
            f"Step {self.simulation.step_count}/{config.generation_steps}  "
            f"Alive {self.simulation.alive_count}/{config.population_size}  "
            f"Best {best.score:.1f}"
        )

    def _toggle_running(self) -> None:
        self.running = not self.running
        self.toggle_button.configure(text="Pause" if self.running else "Resume")

    def _reset(self) -> None:
        self.simulation.reset()

    def _world_coordinates(self, event: tk.Event) -> tuple[float, float]:
        return (
            event.x / max(self.canvas.winfo_width(), 1) * self.simulation.config.width,
            event.y
            / max(self.canvas.winfo_height(), 1)
            * self.simulation.config.height,
        )

    def _add_resource(self, event: tk.Event) -> None:
        self.simulation.add_resource(*self._world_coordinates(event))

    def _add_hazard(self, event: tk.Event) -> None:
        self.simulation.add_hazard(*self._world_coordinates(event))
