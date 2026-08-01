r"""Discrete active-inference agent — Grid World planning (book §9.4–9.5, Alg. 9.5.1).

Assembles the Chapter 9 pieces into a planning agent: at each step it infers its state,
enumerates candidate policies (action sequences), scores each by **expected free energy**,
forms a policy posterior ``σ(−γG)``, marginalizes it to an action posterior, acts, and
repeats (receding horizon). On a Grid World with a one-hot preference at the goal square,
the agent navigates to the goal — *planning as inference*.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.pomdp import (
    FactorialPOMDP,
    HierarchicalPOMDP,
    POMDPModel,
    accumulate_a_counts,
    accumulate_b_counts,
    action_posterior,
    efe_with_novelty,
    evaluate_policy,
    expected_A,
    expected_B,
    expected_D,
    factorial_efe,
    factorial_infer_states,
    hierarchical_layer_vfe,
    infer_states,
    one_hot,
    parameter_novelty,
    policy_posterior,
    policy_posterior_full,
    predict_state,
    softmax,
)

__all__ = [
    "make_gridworld",
    "enumerate_policies",
    "GridWorldResult",
    "simulate_pomdp_agent",
    # §10.1 — learning the POMDP parameters
    "DirichletParameters",
    "LearningResult",
    "learn_D_vector",
    "simulate_array_learning",
    "LearningAgentResult",
    "simulate_learning_agent",
    # §10.2 — precision / habit policy optimization
    "precision_policy_sweep",
    # §10.3 — factorial depth (two-armed bandit)
    "make_two_armed_bandit",
    "TwoArmedBanditResult",
    "simulate_two_armed_bandit",
    # §10.4 — hierarchical depth
    "HierarchicalResult",
    "simulate_hierarchical_agent",
]

# Action set for Grid World: 0=up, 1=down, 2=left, 3=right.
_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def make_gridworld(rows: int, cols: int, goal: int) -> POMDPModel:
    r"""Build a deterministic, fully-observable Grid World POMDP (book §9.4).

    ``rows×cols`` cells indexed row-major. The likelihood ``A`` is the identity (the agent
    observes its own cell), the four ``B[u]`` matrices encode deterministic moves clamped at
    the walls, and the preference ``C`` is one-hot on the ``goal`` cell.
    """
    n = rows * cols
    A = np.eye(n)
    B = np.zeros((4, n, n))
    for s in range(n):
        r, c = divmod(s, cols)
        for u, (dr, dc) in _MOVES.items():
            nr = min(max(r + dr, 0), rows - 1)
            nc = min(max(c + dc, 0), cols - 1)
            B[u, nr * cols + nc, s] = 1.0      # column s → next cell (clamped)
    C = one_hot(goal, n)
    D = one_hot(0, n)
    return POMDPModel(A=A, D=D, B=B, C=C)


def enumerate_policies(n_controls: int, horizon: int) -> List[List[int]]:
    """All action sequences of length ``horizon`` over ``n_controls`` actions."""
    return [list(p) for p in itertools.product(range(n_controls), repeat=horizon)]


@dataclass(frozen=True)
class GridWorldResult:
    """Store discrete grid-world states, actions, beliefs, goal status, and timing."""

    states: np.ndarray         # (T+1,) true cell index over time
    actions: np.ndarray        # (T,) action taken each step
    beliefs: np.ndarray        # (T+1, C) posterior over states each step
    goal: int
    reached: bool
    n_steps_to_goal: Optional[int]


def simulate_pomdp_agent(
    model: POMDPModel,
    *,
    start: int = 0,
    horizon: int = 2,
    gamma: float = 4.0,
    max_steps: int = 12,
    rng: Optional[np.random.Generator] = None,
) -> GridWorldResult:
    r"""Run the discrete active-inference loop (Algorithm 9.5.1, receding horizon).

    Each step: infer the state from the observation, score every length-``horizon`` policy by
    expected free energy, pick the lowest-EFE action via ``σ(−γG)`` marginalized to actions,
    apply it (deterministic transition), and repeat until the goal observation is reached.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_controls = int(model.B.shape[0])
    C = np.asarray(model.C, dtype=float)
    policies = enumerate_policies(n_controls, horizon)
    V = np.asarray(policies)                      # (P, horizon) action at each step
    goal = int(np.argmax(C))

    state = int(start)
    belief = infer_states(model, state)           # observe own cell (A = identity)
    states = [state]
    beliefs = [belief]
    actions: List[int] = []
    reached = state == goal
    n_to_goal = 0 if reached else None

    for t in range(max_steps):
        if reached:
            break
        G = np.array([evaluate_policy(model, belief, pol, C) for pol in policies])
        q_pi = policy_posterior(G, gamma=gamma)
        q_u = action_posterior(q_pi, V, n_controls, tau=0)
        u = int(np.argmax(q_u))
        actions.append(u)
        # Deterministic environment transition (B[u] is a permutation/clamp).
        state = int(np.argmax(model.B[u][:, state]))
        belief = infer_states(model, state, prior=predict_state(model, belief, u))
        states.append(state)
        beliefs.append(belief)
        if state == goal:
            reached = True
            n_to_goal = t + 1

    return GridWorldResult(states=np.array(states), actions=np.array(actions),
                           beliefs=np.array(beliefs), goal=goal, reached=reached,
                           n_steps_to_goal=n_to_goal)


# ---------------------------------------------------------------------------
# §10.1 — Learning the POMDP parameters over trials (Dirichlet, Alg. 10.1.1)
# ---------------------------------------------------------------------------


@dataclass
class DirichletParameters:
    r"""Concentration parameters (pseudocounts) for the learnable POMDP arrays (book §10.1).

    Holds the Dirichlet concentration parameters ``a`` (for the likelihood ``A``), ``b``
    (for the transition ``B``; ``(C, C)`` or ``(U, C, C)``), and ``d`` (for the state prior
    ``D``). The maximum-a-posteriori / expected probability arrays are the Dirichlet means,
    available via :attr:`A`, :attr:`B`, :attr:`D`. ``b`` is optional (omit when transitions
    are not being learned).
    """

    a: np.ndarray
    d: np.ndarray
    b: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)
        self.d = np.asarray(self.d, dtype=float)
        if self.b is not None:
            self.b = np.asarray(self.b, dtype=float)

    @property
    def A(self) -> np.ndarray:
        """Expected likelihood ``A = E[Dir(a)]`` (column-normalized pseudocounts)."""
        return expected_A(self.a)

    @property
    def B(self) -> Optional[np.ndarray]:
        """Expected transition ``B = E[Dir(b)]`` (or ``None`` if ``b`` is not set)."""
        return None if self.b is None else expected_B(self.b)

    @property
    def D(self) -> np.ndarray:
        """Expected state prior ``D = E[Dir(d)]``."""
        return expected_D(self.d)

    def copy(self) -> "DirichletParameters":
        """Return an independent copy of the mutable learning state."""
        return DirichletParameters(
            a=self.a.copy(), d=self.d.copy(),
            b=None if self.b is None else self.b.copy(),
        )


@dataclass(frozen=True)
class LearningResult:
    """Per-trial trace of Dirichlet parameter learning (Examples 10.1–10.3)."""

    D_history: np.ndarray          # (n_trials+1, C) expected D after each trial
    A_history: np.ndarray          # (n_trials+1, O, C) expected A after each trial
    B_history: Optional[np.ndarray]  # (n_trials+1, ...) expected B, or None
    a_confidence: np.ndarray       # (n_trials+1, O, C) likelihood pseudocounts (raw a)
    b_confidence: Optional[np.ndarray]  # (n_trials+1, ...) transition pseudocounts, or None
    d_final: np.ndarray            # (C,) final d concentration parameters
    A_true: Optional[np.ndarray]
    B_true: Optional[np.ndarray]
    D_true: Optional[np.ndarray]

    def final_A_error(self) -> Optional[float]:
        """Max-abs error of the learned ``A`` vs the true ``A`` (``None`` if no ground truth)."""
        if self.A_true is None:
            return None
        return float(np.max(np.abs(self.A_history[-1] - self.A_true)))


def learn_D_vector(
    d0: np.ndarray,
    initial_state_beliefs: "np.ndarray | list",
) -> LearningResult:
    r"""Learn the state-prior vector ``D`` by accumulating initial-state beliefs (Example 10.1).

    Implements the ``d`` update (book Eq. 4c/7) ``d ← d + s^{(0)}`` once per trial. Starting
    from ``d0`` and feeding ``s^{(0)}`` for each trial, the running expected value
    ``D = d / Σd`` converges to the proportions of the supplied beliefs.

    With ``d0 = [1, 1]`` and ``s^{(0)} = [0.9, 0.1]`` repeated for 49 trials, this reproduces
    the book's ``d = [45.1, 5.9]`` and ``D ≈ [0.884, 0.116]`` (Eq. 8).

    Args:
        d0: ``(C,)`` initial concentration parameters.
        initial_state_beliefs: ``(n_trials, C)`` array (or list) of ``s^{(0)}`` per trial.

    Returns:
        :class:`LearningResult` with ``D_history`` of shape ``(n_trials+1, C)``.
    """
    d = np.asarray(d0, dtype=float).copy()
    beliefs = np.atleast_2d(np.asarray(initial_state_beliefs, dtype=float))
    D_hist = [d / d.sum()]
    for s0 in beliefs:
        d = d + s0
        D_hist.append(d / d.sum())
    C = d.shape[0]
    return LearningResult(
        D_history=np.array(D_hist),
        A_history=np.zeros((len(D_hist), 0, C)),
        B_history=None,
        a_confidence=np.zeros((len(D_hist), 0, C)),
        b_confidence=None,
        d_final=d,
        A_true=None, B_true=None, D_true=None,
    )


def simulate_array_learning(
    *,
    A_true: np.ndarray,
    B_true: np.ndarray,
    n_trials: int = 5,
    steps_per_trial: int = 1000,
    learn: str = "A",
    a0: float = 1.0,
    b0: float = 1.0,
    start_state: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> LearningResult:
    r"""Learn ``A`` or ``B`` by counting state–observation / state–state pairs (Examples 10.2–10.3).

    Reproduces the book's likelihood/transition learning experiments (Figs 10.1.3/10.1.4).
    The agent is assumed to have **perfect state knowledge** (so no state estimation is
    needed); on each trial it rolls a Markov chain forward under the true transition
    ``B_true``, emits observations under the true likelihood ``A_true``, and accumulates the
    Dirichlet evidence:

    * ``learn="A"``: ``a += Σ_τ o^{(τ)} ∘ s^{(τ)}`` (Eq. 4a), then ``A = E[Dir(a)]``;
    * ``learn="B"``: ``b += Σ_τ s^{(τ)} ∘ s^{(τ-1)}`` (Eq. 4b), then ``B = E[Dir(b)]``.

    With enough counts the learned array converges on the true proportions, and the
    pseudocounts (confidence) grow linearly with the number of observed pairs.

    Args:
        A_true: ``(O, C)`` true likelihood used by the generative process.
        B_true: ``(C, C)`` true (single, action-free) transition matrix.
        n_trials: number of trials.
        steps_per_trial: time steps ``T`` per trial.
        learn: ``"A"`` or ``"B"`` — which array to learn.
        a0, b0: uniform initial pseudocount for every entry of ``a`` / ``b``.
        start_state: index of the state each trial starts in.
        rng: optional NumPy generator (defaults to seed 0 for reproducibility).

    Returns:
        :class:`LearningResult` with per-trial ``A_history`` / ``B_history`` and confidences.
    """
    if learn not in ("A", "B"):
        raise ValueError("learn must be 'A' or 'B'")
    if rng is None:
        rng = np.random.default_rng(0)
    A_true = np.asarray(A_true, dtype=float)
    B_true = np.asarray(B_true, dtype=float)
    n_obs, C = A_true.shape

    a = np.full((n_obs, C), float(a0))
    b = np.full((C, C), float(b0))

    A_hist = [expected_A(a)]
    B_hist = [expected_B(b)]
    a_conf = [a.copy()]
    b_conf = [b.copy()]

    for _ in range(int(n_trials)):
        s = int(start_state)
        for _t in range(int(steps_per_trial)):
            # Generative process: emit an observation, then (for B) transition.
            o = int(rng.choice(n_obs, p=A_true[:, s]))
            if learn == "A":
                a = a + accumulate_a_counts(one_hot(o, n_obs), one_hot(s, C))
            s_next = int(rng.choice(C, p=B_true[:, s]))
            if learn == "B":
                # Count the transition s -> s_next the moment it occurs (Eq. 4b), so the
                # final transition of each trial is included rather than dropped.
                b = b + accumulate_b_counts(one_hot(s_next, C), one_hot(s, C))
            s = s_next
        A_hist.append(expected_A(a))
        B_hist.append(expected_B(b))
        a_conf.append(a.copy())
        b_conf.append(b.copy())

    learning_B = learn == "B"
    return LearningResult(
        D_history=np.zeros((len(A_hist), C)),
        A_history=np.array(A_hist),
        B_history=np.array(B_hist) if learning_B else None,
        a_confidence=np.array(a_conf),
        b_confidence=np.array(b_conf) if learning_B else None,
        d_final=np.zeros(C),
        A_true=A_true if learn == "A" else None,
        B_true=B_true if learning_B else None,
        D_true=None,
    )


@dataclass(frozen=True)
class LearningAgentResult:
    """Store Algorithm 10.1.1 Dirichlet learning histories, confidence, novelty, and truth."""

    A_history: np.ndarray          # (n_trials+1, O, C) expected A after each trial
    a_confidence: np.ndarray       # (n_trials+1, O, C) raw pseudocounts
    novelty_first_trial: float     # parameter-novelty of the start state on trial 0
    A_true: np.ndarray

    def final_A_error(self) -> float:
        """Return the final value of the stored inference trajectory."""
        return float(np.max(np.abs(self.A_history[-1] - self.A_true)))


def simulate_learning_agent(
    *,
    A_true: np.ndarray,
    n_states: int,
    n_trials: int = 30,
    steps_per_trial: int = 20,
    a0: float = 1.0,
    gamma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> LearningAgentResult:
    r"""Discrete active inference **with learning** — Algorithm 10.1.1 (novelty-driven).

    A compact end-to-end demonstration of learning the likelihood ``A`` while acting. The
    agent starts with uniform pseudocounts (knowing nothing about ``A``) and, on each trial,
    uses the **novelty-augmented** expected free energy :func:`efe_with_novelty` to choose
    which state to sample next — preferring state–observation pairs it is least confident
    about (parameter information-seeking). It then accumulates ``a`` evidence from the
    resulting observation and, at the end of each trial, updates ``A = E[Dir(a)]``.

    Actions move the agent between ``n_states`` states (action ``u`` jumps to state ``u``),
    so the agent can freely choose which mapping to probe. Over trials the novelty term
    decays as pseudocounts grow and the learned ``A`` converges on ``A_true``.

    Returns:
        :class:`LearningAgentResult` with the per-trial ``A_history`` and the trial-0 novelty.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    A_true = np.asarray(A_true, dtype=float)
    n_obs, C = A_true.shape
    if C != n_states:
        raise ValueError("A_true must have n_states columns")

    a = np.full((n_obs, C), float(a0))
    # Uniform preferences (no reward-seeking) so behaviour is novelty-driven.
    C_pref = np.full(n_obs, 1.0 / n_obs)

    A_hist = [expected_A(a)]
    a_conf = [a.copy()]
    novelty0 = parameter_novelty(a, one_hot(0, C))

    for _ in range(int(n_trials)):
        for _t in range(int(steps_per_trial)):
            # Score each reachable state by novelty-augmented EFE (lower = preferred).
            model = POMDPModel(A=expected_A(a), D=np.full(C, 1.0 / C))
            G = np.array([
                efe_with_novelty(model, one_hot(s, C), C_pref, a) for s in range(C)
            ])
            q = policy_posterior(G, gamma=gamma)
            s = int(rng.choice(C, p=q))
            o = int(rng.choice(n_obs, p=A_true[:, s]))
            a = a + accumulate_a_counts(one_hot(o, n_obs), one_hot(s, C))
        A_hist.append(expected_A(a))
        a_conf.append(a.copy())

    return LearningAgentResult(
        A_history=np.array(A_hist),
        a_confidence=np.array(a_conf),
        novelty_first_trial=float(novelty0),
        A_true=A_true,
    )


# ---------------------------------------------------------------------------
# §10.2 — precision / habit policy optimization (Example 10.5)
# ---------------------------------------------------------------------------


def precision_policy_sweep(
    G: np.ndarray,
    gammas: "np.ndarray | list",
    *,
    E: Optional[np.ndarray] = None,
    F: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Policy posterior across a range of precisions ``γ`` (book Example 10.5, Fig 10.2.2).

    For each ``γ`` in ``gammas`` returns the full policy posterior
    ``σ(log E − F − γ G)`` (:func:`policy_posterior_full`). With uniform ``E`` and ``F = None``
    this reproduces the book's precision sweep: at ``γ = 0`` the distribution is the (uniform
    or habit-shaped) prior, and as ``γ`` grows it concentrates on the lowest-EFE policy.

    Args:
        G: ``(P,)`` expected free energy per policy.
        gammas: ``(K,)`` precision values to sweep.
        E: optional ``(P,)`` baseline prior / habit (uniform if ``None``).
        F: optional ``(P,)`` policy-dependent variational free energy.

    Returns:
        ``(K, P)`` array; row ``k`` is the policy posterior at ``gammas[k]``.
    """
    gammas = np.asarray(gammas, dtype=float)
    return np.array([policy_posterior_full(G, F=F, E=E, gamma=float(g)) for g in gammas])


# ---------------------------------------------------------------------------
# §10.3 — Factorial depth: the two-armed bandit task (Example 10.7)
# ---------------------------------------------------------------------------
#
# The two-armed bandit (TAB) is the canonical factorial active-inference task. Two
# hidden state factors — a fixed **context** (which machine pays better) and a
# controllable **choice** — jointly drive three observation modalities (a hint, a
# reward, and a proprioceptive echo of the choice). The agent can sample a *hint*
# (epistemic, resolves the context) or pull a machine (pragmatic, may win). It trades
# off exploration against exploitation purely by minimizing expected free energy.

# Choice-factor states / actions: 0=start, 1=hint, 2=left machine, 3=right machine.
_TAB_START, _TAB_HINT, _TAB_LEFT, _TAB_RIGHT = 0, 1, 2, 3


def make_two_armed_bandit(
    *,
    p_win: float = 0.9,
    hint_reliability: float = 1.0,
    reward_prefs: "tuple[float, float, float]" = (0.0, -3.0, 4.0),
) -> FactorialPOMDP:
    r"""Build the two-armed bandit factorial POMDP (book Example 10.7, Figs 10.3.3–5).

    State factors (``N = 2``): **context** ``{left-better, right-better}`` (fixed, identity
    transition) and **choice** ``{start, hint, left, right}`` (controllable). Observation
    modalities (``M = 3``): **hint** ``{none, left-hint, right-hint}``, **reward**
    ``{start, lose, win}``, **choice** ``{start, hint, left, right}`` (proprioceptive echo).

    Args:
        p_win: payout probability of the *better* machine (the worse machine pays ``1−p_win``).
        hint_reliability: probability the hint points at the truly-better machine.
        reward_prefs: raw preferences over ``(start, lose, win)``; **softmax-normalized** into
            ``C[1]`` (book p. 620: the preference vector "must be softmax-normalized"). The
            default ``(0, −3, 4)`` is a risk-averse agent; ``(0, 0, 4)`` is less risk-averse.

    Returns:
        A :class:`~active_inference.core.pomdp.FactorialPOMDP` with ``A``, ``B``, ``C``, ``D`` set.
    """
    n_ctx, n_choice = 2, 4
    # --- A[0] hint: (3 obs, context, choice) ---------------------------------
    A_hint = np.zeros((3, n_ctx, n_choice))
    for c in range(n_ctx):
        for ch in range(n_choice):
            if ch == _TAB_HINT:                       # only the hint action yields a hint
                # correct hint points at the better machine; reliability splits the mass
                correct = 1 if c == 0 else 2          # ctx0→left-hint(1), ctx1→right-hint(2)
                wrong = 2 if c == 0 else 1
                A_hint[correct, c, ch] = hint_reliability
                A_hint[wrong, c, ch] = 1.0 - hint_reliability
            else:
                A_hint[0, c, ch] = 1.0                # "no hint"
    # --- A[1] reward: (3 obs {start,lose,win}, context, choice) ---------------
    A_rew = np.zeros((3, n_ctx, n_choice))
    for c in range(n_ctx):
        for ch in range(n_choice):
            if ch in (_TAB_START, _TAB_HINT):
                A_rew[0, c, ch] = 1.0                 # "start" (no reward yet)
            else:
                machine = 0 if ch == _TAB_LEFT else 1
                win_p = p_win if machine == c else (1.0 - p_win)
                A_rew[2, c, ch] = win_p               # win
                A_rew[1, c, ch] = 1.0 - win_p         # lose
    # --- A[2] choice echo: (4 obs, context, choice) = identity on choice ------
    A_choice = np.zeros((n_choice, n_ctx, n_choice))
    for c in range(n_ctx):
        for ch in range(n_choice):
            A_choice[ch, c, ch] = 1.0
    # --- B ---------------------------------------------------------------------
    B_ctx = np.eye(n_ctx)[:, :, None]                 # context cannot change (1 null action)
    B_choice = np.zeros((n_choice, n_choice, n_choice))
    for u in range(n_choice):
        B_choice[u, :, u] = 1.0                       # action u → next choice = u (deterministic)
    # --- C / D -----------------------------------------------------------------
    C = [np.ones(3) / 3.0, softmax(np.asarray(reward_prefs, dtype=float)), np.ones(n_choice) / n_choice]
    D = [np.array([0.5, 0.5]), one_hot(_TAB_START, n_choice)]
    return FactorialPOMDP(A=[A_hint, A_rew, A_choice], D=D, B=[B_ctx, B_choice], C=C)


@dataclass(frozen=True)
class TwoArmedBanditResult:
    """Store bandit context beliefs, choices, rewards, policy posteriors, and wins."""

    context_belief: np.ndarray     # (T+1, 2) belief over {left-better, right-better}
    choices: np.ndarray            # (T,) choice action each step
    reward_obs: np.ndarray         # (T,) reward observation index {0 start,1 lose,2 win}
    policy_posterior: np.ndarray   # (T, n_actions) action posterior each step
    true_context: int
    n_wins: int
    n_hints: int


def simulate_two_armed_bandit(
    model: FactorialPOMDP,
    *,
    true_context: int = 1,
    n_steps: int = 12,
    gamma: float = 4.0,
    rng: "np.random.Generator | None" = None,
) -> TwoArmedBanditResult:
    r"""Run the two-armed bandit agent (book Example 10.7, Figs 10.3.6/10.3.7).

    A receding-horizon loop: each step the agent infers both state factors (the context
    belief is carried forward and refined by any hint), scores every one-step choice action
    by **factorial expected free energy** ``G = Σ_m risk_m + ambiguity_m`` (Eq. 38), forms the
    policy posterior ``σ(−γG)``, acts, and samples observations from the true environment.

    The epistemic *hint* action has low ambiguity (it is informative) and resolves the
    context; the pragmatic *machine* actions realize the reward preference once the context
    is known. The interplay reproduces the explore-then-exploit behaviour of the book figures.

    Args:
        model: a :func:`make_two_armed_bandit` model.
        true_context: the actual better machine (0 = left, 1 = right).
        n_steps: number of environment steps.
        gamma: policy precision.
        rng: optional NumPy generator (seed 0 by default).

    Returns:
        :class:`TwoArmedBanditResult` with the per-step context belief, choices, rewards,
        and policy posteriors.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_actions = model.factor_sizes[1]
    ctx_belief = np.array([0.5, 0.5])
    choice_state = _TAB_START

    ctx_hist = [ctx_belief.copy()]
    choices: List[int] = []
    rewards: List[int] = []
    pol_hist: List[np.ndarray] = []
    n_wins = n_hints = 0

    for _ in range(int(n_steps)):
        # Score each next choice action by one-step factorial EFE from the current belief.
        G = np.empty(n_actions)
        for u in range(n_actions):
            pred_ctx = ctx_belief                       # context is fixed (identity B)
            pred_choice = model.B[1][:, choice_state, u]
            G[u] = factorial_efe(model, [pred_ctx, pred_choice])
        q_pi = policy_posterior(G, gamma=gamma)
        pol_hist.append(q_pi.copy())
        u = int(np.argmax(q_pi))
        choices.append(u)
        if u == _TAB_HINT:
            n_hints += 1

        # Environment: choice transitions deterministically; sample observations.
        choice_state = u
        obs = [int(rng.choice(model.obs_sizes[m],
                              p=model.A[m][:, true_context, choice_state]))
               for m in range(model.n_modalities)]
        if obs[1] == 2:
            n_wins += 1
        rewards.append(obs[1])

        # Belief update: context carried (identity), choice = realized action (one-hot prior).
        priors = [ctx_belief, one_hot(choice_state, n_actions)]
        post = factorial_infer_states(model, obs, priors=priors)
        ctx_belief = post[0]
        ctx_hist.append(ctx_belief.copy())

    return TwoArmedBanditResult(
        context_belief=np.array(ctx_hist), choices=np.array(choices),
        reward_obs=np.array(rewards), policy_posterior=np.array(pol_hist),
        true_context=int(true_context), n_wins=n_wins, n_hints=n_hints,
    )


# ---------------------------------------------------------------------------
# §10.4 — Hierarchical depth: nested-timescale simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchicalResult:
    """Trace of :func:`simulate_hierarchical_agent` (book §10.4, nested time scales)."""

    top_belief: np.ndarray          # (n_macro+1, C_top) top-layer belief over macro-steps
    bottom_priors: np.ndarray       # (n_macro, C_bottom) top-down prior into the bottom layer
    bottom_belief: np.ndarray       # (n_macro, inner_steps, C_bottom) bottom-layer beliefs
    layer_vfe: np.ndarray           # (n_macro,) total bottom-layer VFE per macro-step
    true_top: int


def simulate_hierarchical_agent(
    model: HierarchicalPOMDP,
    *,
    true_top: int = 0,
    bottom_observations: "list[list[int]] | None" = None,
    n_macro: int = 4,
    inner_steps: int = 3,
    rng: "np.random.Generator | None" = None,
) -> HierarchicalResult:
    r"""Run a two-layer hierarchical POMDP over nested time scales (book §10.4, Eq. 39–50).

    Demonstrates the defining hierarchical mechanism: the **slow** top layer's state sets the
    **fast** bottom layer's initial-state prior through the link map (Eq. 42,
    ``prior = link · s^{[l+1]}``). Each macro-step the top layer filters its (slow)
    observation; its belief is pushed down to seed the bottom layer, which then filters
    ``inner_steps`` fast observations. The bottom-layer posterior is therefore *contextualized*
    by the top layer — the core of nested-timescale generative models.

    Args:
        model: a 2-layer :class:`~active_inference.core.pomdp.HierarchicalPOMDP` with a link.
        true_top: the true top-layer state (held fixed; it is the slow context).
        bottom_observations: optional ``n_macro × inner_steps`` bottom observation indices; if
            ``None`` they are sampled from the bottom layer's ``A`` at the linked prior's argmax.
        n_macro: number of macro (top-layer) steps.
        inner_steps: number of fast (bottom-layer) steps per macro-step.
        rng: optional NumPy generator.

    Returns:
        :class:`HierarchicalResult` with per-macro-step top belief, top-down priors, bottom
        beliefs, and bottom-layer VFE.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if model.n_layers != 2:
        raise ValueError("simulate_hierarchical_agent expects a 2-layer model")
    top, bottom = model.layers[1], model.layers[0]
    top_belief = np.asarray(top.D, dtype=float).copy()

    top_hist = [top_belief.copy()]
    priors_hist: List[np.ndarray] = []
    bottom_hist: List[np.ndarray] = []
    vfe_hist: List[float] = []

    for k in range(int(n_macro)):
        # Slow layer: observe the (fixed) top state and filter.
        top_obs = int(rng.choice(top.n_obs, p=top.A[:, true_top]))
        top_belief = infer_states(top, top_obs, prior=top_belief)
        top_hist.append(top_belief.copy())

        # Top-down: the bottom layer's prior is the linked top belief (Eq. 42).
        prior = model.layer_prior(0, top_belief)
        priors_hist.append(prior.copy())

        # Fast layer: filter inner_steps observations from that prior.
        belief = prior.copy()
        inner_beliefs, vfe_total = [], 0.0
        for t in range(int(inner_steps)):
            if bottom_observations is not None:
                o = int(bottom_observations[k][t])
            else:
                o = int(rng.choice(bottom.n_obs, p=bottom.A[:, int(np.argmax(prior))]))
            # VFE is a divergence between the new posterior and the *previous*
            # predictive prior. Capture the pre-update belief so the KL term in
            # hierarchical_layer_vfe (log s − log prior) stays non-zero instead of
            # collapsing when prior is set to the freshly-updated posterior.
            prior_t = belief
            belief = infer_states(bottom, o, prior=prior_t)
            vfe_total += hierarchical_layer_vfe(belief, o, bottom.A, prior=prior_t)
            inner_beliefs.append(belief.copy())
        bottom_hist.append(np.array(inner_beliefs))
        vfe_hist.append(vfe_total)

    return HierarchicalResult(
        top_belief=np.array(top_hist), bottom_priors=np.array(priors_hist),
        bottom_belief=np.array(bottom_hist), layer_vfe=np.array(vfe_hist),
        true_top=int(true_top),
    )
