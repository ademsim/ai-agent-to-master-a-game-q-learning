import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Taxi Q-Learning Agent")

st.title("🚕 Taxi: an AI Agent that Learns by Trial and Error")
st.write(
    "A Q-learning agent learns to play the Taxi game (a classic from the Gymnasium library) without knowing the rules. "
    "The taxi must pick up the passenger (blue dot) and drop them at the pink square. "
    "Every step costs -1, a correct drop-off gives +20, and an illegal pick-up or drop-off costs -10."
)

MAP = ["+---------+", "|R: | : :G|", "| : | : : |", "| : : : : |", "| | : | : |", "|Y| : |B: |", "+---------+"]
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]
STOP_NAMES = ["R", "G", "Y", "B"]
ACTIONS = ["south", "north", "east", "west", "pickup", "dropoff"]
STOP_COLORS = ["#d62728", "#2ca02c", "#bcbd22", "#1f77b4"]


def build_game():
    """The rules of the Taxi game (the same as Gymnasium's Taxi): 500 states, 6 actions, no random moves."""
    ns, rew, done = np.zeros((500, 6), int), np.full((500, 6), -1.0), np.zeros((500, 6), bool)
    starts = []
    for row in range(5):
        for col in range(5):
            for p in range(5):
                for d in range(4):
                    s = ((row * 5 + col) * 5 + p) * 4 + d
                    if p < 4 and p != d:
                        starts.append(s)
                    for a in range(6):
                        nr, nc, npass, r, term = row, col, p, -1.0, False
                        if a == 0:
                            nr = min(row + 1, 4)
                        elif a == 1:
                            nr = max(row - 1, 0)
                        elif a == 2 and col < 4 and MAP[row + 1][2 * col + 2] != "|":
                            nc = col + 1
                        elif a == 3 and col > 0 and MAP[row + 1][2 * col] != "|":
                            nc = col - 1
                        elif a == 4:
                            if p < 4 and (row, col) == LOCS[p]:
                                npass = 4
                            else:
                                r = -10.0
                        elif a == 5:
                            if (row, col) == LOCS[d] and p == 4:
                                npass, term, r = d, True, 20.0
                            elif (row, col) in LOCS and p == 4:
                                npass = LOCS.index((row, col))
                            else:
                                r = -10.0
                        ns[s, a], rew[s, a], done[s, a] = ((nr * 5 + nc) * 5 + npass) * 4 + d, r, term
    return ns, rew, done, np.array(starts)


with st.spinner("Loading the game..."):
    NS, REW, DONE, STARTS = build_game()


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
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

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
    if passenger < 4:
        pr, pc = LOCS[passenger]
        ax.add_patch(Circle((pc + 0.5, pr + 0.62), 0.16, color="#1f77b4"))
    color = "#2ca02c" if passenger == 4 else "#ffdd00"
    ax.add_patch(Rectangle((col + 0.2, row + 0.3), 0.6, 0.4, color=color, ec="black", lw=1.5))
    ax.text(col + 0.5, row + 0.5, "taxi", ha="center", va="center", fontsize=8)
    return fig


with st.form("settings"):
    c1, c2 = st.columns(2)
    episodes = c1.select_slider("Training games", [200, 500, 1000, 2000, 5000, 10000], value=5000)
    alpha = c2.select_slider("Learning rate (alpha)", [0.01, 0.05, 0.1, 0.3, 0.5, 1.0], value=0.1)
    gamma = c1.select_slider("Discount factor (gamma)", [0.5, 0.8, 0.9, 0.95, 0.99], value=0.99)
    seed = c2.number_input("Random seed", 0, 9999, 42)
    submitted = st.form_submit_button("Train the agent")

if submitted:
    with st.spinner("Training..."):
        Q, history = 
