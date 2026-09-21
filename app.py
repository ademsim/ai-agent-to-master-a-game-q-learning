import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle, Rectangle

st.set_page_config(page_title="Taxi Q-Learning Agent")

MAP = ["+---------+", "|R: | : :G|", "| : | : : |", "| : : : : |", "| | : | : |", "|Y| : |B: |", "+---------+"]
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]
STOP_NAMES = ["R", "G", "Y", "B"]
ACTIONS = ["south", "north", "east", "west", "pickup", "dropoff"]
STOP_COLORS = ["#d62728", "#2ca02c", "#bcbd22", "#1f77b4"]


@st.cache_resource
def load_game():
    name = "Taxi-v4" if "Taxi-v4" in gym.registry else "Taxi-v3"
    u = gym.make(name).unwrapped
    ns, rew, done = np.zeros((500, 6), int), np.zeros((500, 6)), np.zeros((500, 6), bool)
    for s in range(500):
        for a in range(6):
            _, ns[s, a], rew[s, a], done[s, a] = u.P[s][a][0]
    starts = np.flatnonzero(u.initial_state_distrib)
    return ns, rew, done, starts


NS, REW, DONE, STARTS = load_game()


def decode(state):
    dest = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    return state // 5, state % 5, passenger, dest


def play(policy, start, rng=None, max_steps=200):
    """Play one game. Returns the states, actions, rewards, and whether the passenger was delivered."""
    states, actions, rewards, delivered = [start], [], [], False
    s = start
    for _ in range(max_steps):
        a = int(rng.integers(6)) if policy is None else int(policy[s])
        actions.append(a)
        rewards.append(REW[s, a])
        delivered = bool(DONE[s, a])
        s = NS[s, a]
        states.append(s)
        if delivered:
            break
    return states, actions, rewards, delivered


@st.cache_data
def train(episodes, alpha, gamma, decay, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((500, 6))
    eps, history = 1.0, []
    for _ in range(episodes):
        s, total = int(rng.choice(STARTS)), 0.0
        for _ in range(200):
            a = int(rng.integers(6)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, term = NS[s, a], REW[s, a], DONE[s, a]
            Q[s, a] += alpha * (r + gamma * (0 if term else Q[s2].max()) - Q[s, a])
            s, total = s2, total + r
            if term:
                break
        eps = max(0.05, eps * decay)
        history.append(total)
    return Q, np.array(history)


@st.cache_data
def optimal_policy(gamma=0.99):
    V = np.zeros(500)
    for _ in range(2000):
        Qv = REW + gamma * np.where(DONE, 0, V[NS])
        V_new = Qv.max(axis=1)
        if np.abs(V_new - V).max() < 1e-9:
            break
        V = V_new
    return (REW + gamma * np.where(DONE, 0, V[NS])).argmax(axis=1)


@st.cache_data
def evaluate(policy, n=200):
    rng = np.random.default_rng(0)
    starts = np.random.default_rng(123).choice(STARTS, size=n)
    rows = []
    for s0 in starts:
        _, actions, rewards, delivered = play(policy, int(s0), rng)
        rows.append({"reward": sum(rewards), "steps": len(actions), "success": float(delivered)})
    return pd.DataFrame(rows).mean()


def draw_board(state):
    row, col, passenger, dest = decode(state)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    dr, dc = LOCS[dest]
    ax.add_patch(Rectangle((dc, dr), 1, 1, color="#e377c2", alpha=0.35))
    for i, (r, c) in enumerate(LOCS):
        ax.text(c + 0.14, r + 0.22, STOP_NAMES[i], color=STOP_COLORS[i], fontsize=14, fontweight="bold", va="center")
    for x in range(6):
        ax.plot([x, x], [0, 5], color="#cccccc", lw=0.8)
        ax.plot([0, 5], [x, x], color="#cccccc", lw=0.8)
    ax.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], color="black", lw=3)
    for r in range(5):
        for c in range(4):
            if MAP[r + 1][2 * c + 2] == "|":
                ax.plot([c + 1, c + 1], [r, r + 1], color="black", lw=3)
  
