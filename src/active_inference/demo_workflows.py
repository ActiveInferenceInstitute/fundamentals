"""Reusable workflows behind ``demo/`` application orchestrators.

Each builder composes existing ``core/`` and ``estimators/`` APIs into a
pedagogical application demo. Domain-specific physics (oculomotor plants,
bicycle nonholonomy, quadcopter attitude) are intentionally out of scope for
v1; the plots relabel book algorithms with application metaphors.
"""

from __future__ import annotations

import numpy as np

from .core.lgs import LinearGaussianSystem
from .estimators.applications import simulate_fault_tolerant_control, simulate_robot_navigation
from .estimators.active_inference import simulate_multivariate_active_inference
from .estimators.pomdp import enumerate_policies, evaluate_policy, make_gridworld, simulate_pomdp_agent
from .orchestrator_workflows import (
    MULTIVARIATE_AIF_EXOGENOUS,
    MULTIVARIATE_AIF_PREFERENCE,
    WorkflowResult,
    build_multivariate_active_agent_env,
)
from .visualizations import finalize, panel_grid
from .visualizations.style import COLORS, annotate_stat_box


def build_eye_saccades_demo(
    *,
    rows: int = 5,
    cols: int = 5,
    start: int = 12,
    goal: int | None = None,
    horizon: int = 6,
    gamma: float = 8.0,
    max_steps: int = 20,
) -> WorkflowResult:
    """Discrete EFE saccade planning on a retinotopic fixation grid (Ch.9 POMDP)."""
    n_states = rows * cols
    if goal is None:
        goal = n_states - 1
    if not (0 <= start < n_states and 0 <= goal < n_states):
        raise ValueError("start and goal must lie on the grid")
    model = make_gridworld(rows, cols, goal=goal)
    res = simulate_pomdp_agent(
        model,
        start=start,
        horizon=horizon,
        gamma=gamma,
        max_steps=max_steps,
    )
    policies = enumerate_policies(4, horizon)
    belief0 = np.zeros(n_states)
    belief0[start] = 1.0
    best_efe = []
    for action in range(4):
        scores = [
            evaluate_policy(model, belief0, policy, np.asarray(model.C))
            for policy in policies
            if policy[0] == action
        ]
        if not scores:
            raise RuntimeError(
                f"no policy starts with action {action}; cannot plan a first saccade")
        best_efe.append(min(scores))

    action_names = ["up", "down", "left", "right"]
    fig, axes = panel_grid(
        2,
        title="Eye saccades · discrete expected free energy fixation planning",
        figsize=(11, 4.8),
    )
    ax = axes[0]
    grid = np.zeros((rows, cols))
    for step, cell in enumerate(res.states):
        grid[divmod(cell, cols)] = step + 1
    ax.imshow(grid, cmap="Blues", origin="upper")
    for cell in res.states:
        row, col = divmod(cell, cols)
        ax.plot(col, row, "o", color=COLORS["posterior"], ms=8)
    goal_row, goal_col = divmod(goal, cols)
    start_row, start_col = divmod(start, cols)
    ax.plot(goal_col, goal_row, "*", color=COLORS["likelihood"], ms=22, label="salient target")
    ax.plot(start_col, start_row, "s", color=COLORS["state"], ms=11, label="initial fixation")
    annotate_stat_box(
        ax,
        f"reached={res.reached}\nsteps={res.n_steps_to_goal}",
        loc="lower left",
        fontsize=10,
    )
    finalize(ax, xlabel="eccentricity (columns)", ylabel="eccentricity (rows)", title="saccade path", legend=True)

    ax = axes[1]
    colors = [
        COLORS["posterior"] if value == min(best_efe) else COLORS["neutral"]
        for value in best_efe
    ]
    ax.bar(action_names, best_efe, color=colors, alpha=0.85)
    chosen = action_names[int(np.argmin(best_efe))]
    annotate_stat_box(ax, f"first saccade: {chosen}", loc="upper right", fontsize=10)
    finalize(ax, xlabel="saccade direction", ylabel="min G(π)", title="expected free energy", legend=False)

    arrays = {
        "grid_path": np.asarray(res.states, dtype=int),
        "actions": np.asarray(res.actions, dtype=int),
        "best_first_action_efe": np.asarray(best_efe, dtype=float),
        "goal_index": np.array([goal], dtype=int),
        "start_index": np.array([start], dtype=int),
    }
    return WorkflowResult(
        figures={"visualize_eye_saccades": fig},
        arrays=arrays,
        metadata={"rows": rows, "cols": cols, "horizon": horizon, "gamma": gamma},
        summary={"reached": res.reached, "n_steps": res.n_steps_to_goal},
    )


def build_bicycle_demo(
    *,
    seed: int = 7,
    n_steps: int = 400,
    dt: float = 0.01,
    embedding_dim: int = 2,
    gamma: float = 0.5,
) -> WorkflowResult:
    """Balance metaphor via multivariate active inference plus fault compensation (Ch.7, Ch.13)."""
    action_start = n_steps // 5
    agent, env = build_multivariate_active_agent_env(0.5, embedding_dim=embedding_dim, gamma=gamma)
    passive_agent, _ = build_multivariate_active_agent_env(0.0, embedding_dim=embedding_dim, gamma=gamma)
    rng = np.random.default_rng(seed)
    active = simulate_multivariate_active_inference(
        agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((embedding_dim, 2)),
        n_steps=n_steps,
        dt=dt,
        action_start=action_start,
        rng=rng,
    )
    passive = simulate_multivariate_active_inference(
        passive_agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((embedding_dim, 2)),
        n_steps=n_steps,
        dt=dt,
        action_start=action_start,
        # Same seed as the active sim deliberately: identical noise realizations
        # keep the active-vs-passive comparison a clean A/B on action alone.
        rng=np.random.default_rng(seed),
    )
    fault = simulate_fault_tolerant_control(n_steps=90, fault_time=0.45, post_fault_efficacy=0.55)

    fig, axes = panel_grid(
        2,
        title="Riding a bicycle · balance as multivariate active inference",
        figsize=(11, 4.8),
    )
    time = np.arange(active.xs.shape[0]) * dt
    ax = axes[0]
    ax.plot(time, active.xs[:, 0], color=COLORS["posterior"], label="roll (active)")
    ax.plot(time, active.xs[:, 1], color=COLORS["likelihood"], label="lean (active)")
    ax.plot(time, passive.xs[:, 0], color=COLORS["neutral"], ls="--", alpha=0.7, label="roll (passive)")
    ax.axhline(MULTIVARIATE_AIF_PREFERENCE[0], color=COLORS["state"], ls=":", lw=1.2, label="upright preference")
    annotate_stat_box(
        ax,
        f"active err={active.preference_error(MULTIVARIATE_AIF_PREFERENCE):.3f}",
        loc="upper right",
        fontsize=9,
    )
    finalize(ax, xlabel="time (s)", ylabel="state", title="balance stabilization", legend=True)

    ax = axes[1]
    ax.plot(fault.time, fault.desired, color=COLORS["state"], label="desired steering")
    ax.plot(fault.time, fault.actual, color=COLORS["posterior"], label="actual response")
    ax.axvline(0.45, color=COLORS["likelihood"], ls="--", lw=1.0, label="wobble event")
    finalize(ax, xlabel="normalized time", ylabel="control", title="fault-tolerant compensation", legend=True)

    arrays = {
        "time": time,
        "active_xs": active.xs,
        "passive_xs": passive.xs,
        "active_actions": active.actions,
        "active_free_energy": active.free_energies,
        "fault_time": fault.time,
        "fault_desired": fault.desired,
        "fault_actual": fault.actual,
        "preference": MULTIVARIATE_AIF_PREFERENCE,
        "exogenous": MULTIVARIATE_AIF_EXOGENOUS,
    }
    return WorkflowResult(
        figures={"visualize_bicycle": fig},
        arrays=arrays,
        metadata={"seed": seed, "dt": dt, "gamma": gamma, "action_start": action_start},
        summary={
            "active_preference_error": active.preference_error(MULTIVARIATE_AIF_PREFERENCE),
            "passive_preference_error": passive.preference_error(MULTIVARIATE_AIF_PREFERENCE),
        },
    )


def build_drone_flight_demo(
    *,
    seed: int = 11,
    rows: int = 4,
    cols: int = 4,
    start: int = 0,
    goal: int | None = None,
    horizon: int = 6,
    gamma: float = 8.0,
    max_steps: int = 20,
    nav_goal: tuple[float, float] = (1.0, 0.75),
) -> WorkflowResult:
    """Hybrid-lite drone demo: discrete waypoint lattice, smooth path, LGS fusion (Ch.9, Ch.13, Ch.3)."""
    n_states = rows * cols
    if goal is None:
        goal = n_states - 1
    model = make_gridworld(rows, cols, goal=goal)
    plan = simulate_pomdp_agent(
        model,
        start=start,
        horizon=horizon,
        gamma=gamma,
        max_steps=max_steps,
    )
    navigation = simulate_robot_navigation(n_steps=80, goal=nav_goal)

    rng = np.random.default_rng(seed)
    lgs = LinearGaussianSystem(
        Theta=np.eye(2),
        cov_y=np.eye(2) * 0.08,
        mx=np.zeros(2),
        cov_x=np.eye(2) * 1.5,
    )
    noisy_reads = navigation.path + rng.normal(0.0, 0.12, navigation.path.shape)
    posterior = lgs.posterior_batch(noisy_reads)

    fig, axes = panel_grid(
        3,
        title="Drone flight · waypoint planning, executed path, and noisy position fusion",
        figsize=(12, 4.2),
    )
    ax = axes[0]
    grid = np.zeros((rows, cols))
    for step, cell in enumerate(plan.states):
        grid[divmod(cell, cols)] = step + 1
    ax.imshow(grid, cmap="Blues", origin="upper")
    goal_row, goal_col = divmod(goal, cols)
    ax.plot(goal_col, goal_row, "*", color=COLORS["likelihood"], ms=20, label="waypoint")
    finalize(ax, xlabel="lattice column", ylabel="lattice row", title="discrete EFE plan", legend=True)

    ax = axes[1]
    ax.plot(navigation.path[:, 0], navigation.path[:, 1], color=COLORS["posterior"], label="executed path")
    ax.plot(navigation.goal[0], navigation.goal[1], "*", color=COLORS["likelihood"], ms=18, label="hover goal")
    ax.scatter(noisy_reads[:, 0], noisy_reads[:, 1], s=12, color=COLORS["neutral"], alpha=0.5, label="noisy GPS")
    finalize(ax, xlabel="x", ylabel="y", title="kinematic navigation overlay", legend=True)

    ax = axes[2]
    ax.errorbar(
        posterior.mean[0],
        posterior.mean[1],
        xerr=2 * posterior.std()[0],
        yerr=2 * posterior.std()[1],
        fmt="o",
        color=COLORS["posterior"],
        capsize=4,
        label="LGS posterior",
    )
    ax.plot(navigation.goal[0], navigation.goal[1], "*", color=COLORS["likelihood"], ms=18, label="goal")
    annotate_stat_box(
        ax,
        f"mean=({posterior.mean[0]:.2f}, {posterior.mean[1]:.2f})",
        loc="upper left",
        fontsize=9,
    )
    finalize(ax, xlabel="x", ylabel="y", title="position belief from noisy reads", legend=True)

    arrays = {
        "plan_states": np.asarray(plan.states, dtype=int),
        "plan_actions": np.asarray(plan.actions, dtype=int),
        "navigation_path": navigation.path,
        "navigation_goal": navigation.goal,
        "navigation_preference": navigation.preference,
        "noisy_reads": noisy_reads,
        "posterior_mean": posterior.mean,
        "posterior_std": posterior.std(),
    }
    return WorkflowResult(
        figures={"visualize_drone_flight": fig},
        arrays=arrays,
        metadata={"seed": seed, "rows": rows, "cols": cols, "horizon": horizon, "gamma": gamma},
        summary={"plan_reached": plan.reached, "plan_steps": plan.n_steps_to_goal},
    )


__all__ = [
    "build_bicycle_demo",
    "build_drone_flight_demo",
    "build_eye_saccades_demo",
]
