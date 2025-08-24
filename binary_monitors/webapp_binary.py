import streamlit as st
import json
import numpy as np
from scipy.optimize import minimize # type: ignore
import graphviz # type: ignore
from typing import Any, Literal, Optional
from binary_solvers import *
st.set_page_config(layout="wide")

# Single-monitor solvers

"""
Indices reference for x_single:
Here M is just some arbitrary monitor.

x_single[0] = p_M (probability of using a monitor at all)
x_single[1] = p_A|{} (probability of auditing straight up)
x_single[2] = p_A|M
x_single[3] = p_A|-M
"""

def p_a_given_q1_single_monitor(x_single: list[float], prob: dict[str, float], monitor_name: Literal["M", "N"]) -> float:
    """
    Computes P(A=1 | Q=1) for a single monitor.
    """
    return x_single[1] + x_single[0] * (prob[f"p_{monitor_name.lower()}_given_q"] * x_single[2] + (1 - prob[f"p_{monitor_name.lower()}_given_q"]) * x_single[3])

def p_a_given_q0_single_monitor(x_single: list[float], prob: dict[str, float], monitor_name: Literal["M", "N"]) -> float:
    """
    Computes P(A=1 | Q=0) for a single monitor.
    """
    return x_single[1] + x_single[0] * (prob[f"p_{monitor_name.lower()}_given_not_q"] * x_single[2] + (1 - prob[f"p_{monitor_name.lower()}_given_not_q"]) * x_single[3])

def monitor_cost_single_monitor(x_single: list[float], monitor_name: Literal["M", "N"], c_m: Optional[float] = None, c_n: Optional[float] = None) -> float:
    if monitor_name == "M":
        assert c_m is not None, "c_m must be provided if monitor_name is M"
        return c_m*x_single[0]
    else:
        assert c_n is not None, "c_n must be provided if monitor_name is N"
        return c_n*x_single[0]

def total_cost_single_monitor(x_single: list[float],prob: dict[str, float], monitor_name: Literal["M", "N"], c_a: float, c_m: Optional[float] = None, c_n: Optional[float] = None) -> float:
    return monitor_cost_single_monitor(x_single, monitor_name, c_m, c_n) + c_a*p_a_given_q0_single_monitor(x_single,prob, monitor_name)


def sqp_single_monitor(prob: dict[str, float], monitor_name: Literal["M", "N"], c_a: float, Y: float, use_extra_audit_budget: bool = False,
                       T: Optional[float] = None,
                       c_m: Optional[float] = None, c_n: Optional[float] = None, n_starts: int = 10) -> tuple[list[float], float]:
    """
    Solves the single monitor problem using SLSQP.
    """
    x_single = np.random.rand(4)
    if use_extra_audit_budget:
        assert T is not None, "T must be provided if use_extra_audit_budget is True"
        cons = [
            # p_A|{} <= T
            {'type': 'ineq', 'fun': lambda xx: T - p_a_given_q0_single_monitor(xx, prob, monitor_name)},
            # E[C(A)|Q=0] <= Y
            {'type': 'ineq', 'fun': lambda xx: Y - monitor_cost_single_monitor(xx, monitor_name, c_m, c_n)},
            # p_M + p_A|{} <= 1
            {'type': 'ineq', 'fun': lambda xx: 1.0 - (xx[0] + xx[1])},
        ]
    else:
        cons = [
            # E[C(A)|Q=0] <= Y
            {'type': 'ineq', 'fun': lambda xx: Y - total_cost_single_monitor(xx, prob, monitor_name, c_a, c_m, c_n)},
            # p_M + p_A|{} <= 1
            {'type': 'ineq', 'fun': lambda xx: 1.0 - (xx[0] + xx[1])},
        ]
    bounds = [(0.0, 1.0)] * 4
    def objective(x_single):
        return -p_a_given_q1_single_monitor(x_single, prob, monitor_name)
    best_x = None
    best_obj = -np.inf
    for _ in range(n_starts):
        x_single = np.random.rand(4)
        res = minimize(objective, x_single,
                   method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 1000, 'ftol': 1e-9})
        if res.success:
            print(f"Found optimum {-res.fun:2f}")
            if res.fun > best_obj:
                best_x = res.x
                best_obj = -res.fun
    if best_x is None:
        raise ValueError("No feasible solution found")
    
    return best_x, best_obj
    


# Multi-monitor solvers

"""
The 13 variables x are (in order):
      x[0] = p_M
      x[1] = p_{A|M=1}, x[2] = p_{A|M=0}, x[3] = p_{Q|M=1}, x[4] = p_{Q|M=0}
      x[5] = p_{A|N=1}, x[6] = p_{A|N=0}, x[7] = p_{Q|N=1}, x[8] = p_{Q|N=0}
      x[9] = p_{A|M=1,N=1}, x[10] = p_{A|M=1,N=0},
      x[11] = p_{A|M=0,N=1}, x[12] = p_{A|M=0,N=0}.
"""




def create_game_tree(best_x):
    # Create a new directed graph
    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")  # Left to right layout

    # Function to extract probability from label and scale edge width
    def get_penwidth(label):
        try:
            prob_str = label.split(":")[
                -1
            ].strip()  # Handle labels like "A: 0.2f" and "0.2f"
            prob = float(prob_str)
            return str(0.1 + 3 * prob)  # Scale width from 1 (min) to 5 (max)
        except:
            return "1"  # Default width if parsing fails

    # Start node
    dot.node("start", "Start")

    # First level: Monitor choice
    dot.node("M", "M")
    dot.node("N", "N")
    prob_M_label = f"p(M): {best_x[0]:.4f}"
    prob_N_label = f"p(N): {1 - best_x[0]:.4f}"  # Assuming p(N) = 1 - p(M)
    dot.edge("start", "M", prob_M_label, penwidth=get_penwidth(prob_M_label))
    dot.edge("start", "N", prob_N_label, penwidth=get_penwidth(prob_N_label))

    # Second level: Monitor signals
    # For M branch
    dot.node("M1", "M=1")
    dot.node("M0", "M=0")
    dot.edge("M", "M1", penwidth="1")  # Default width for non-probability edges
    dot.edge("M", "M0", penwidth="1")  # Default width for non-probability edges

    # For N branch
    dot.node("N1", "N=1")
    dot.node("N0", "N=0")
    dot.edge("N", "N1", penwidth="1")  # Default width for non-probability edges
    dot.edge("N", "N0", penwidth="1")  # Default width for non-probability edges

    # Third level: Action choices after first signal
    # After M=1
    dot.node("M1_A", "Audit")
    dot.node("M1_Q", "Query N")
    prob_M1_A_label = f"p(A|M=1): {best_x[1]:.4f}"
    prob_M1_Q_label = f"p(Q|M=1): {best_x[3]:.4f}"
    dot.edge("M1", "M1_A", prob_M1_A_label, penwidth=get_penwidth(prob_M1_A_label))
    dot.edge("M1", "M1_Q", prob_M1_Q_label, penwidth=get_penwidth(prob_M1_Q_label))

    # After M=0
    dot.node("M0_A", "Audit")
    dot.node("M0_Q", "Query N")
    prob_M0_A_label = f"p(A|M=0): {best_x[2]:.4f}"
    prob_M0_Q_label = f"p(Q|M=0): {best_x[4]:.4f}"
    dot.edge("M0", "M0_A", prob_M0_A_label, penwidth=get_penwidth(prob_M0_A_label))
    dot.edge("M0", "M0_Q", prob_M0_Q_label, penwidth=get_penwidth(prob_M0_Q_label))

    # After N=1
    dot.node("N1_A", "Audit")
    dot.node("N1_Q", "Query M")
    prob_N1_A_label = f"p(A|N=1): {best_x[5]:.4f}"
    prob_N1_Q_label = f"p(Q|N=1): {best_x[7]:.4f}"
    dot.edge("N1", "N1_A", prob_N1_A_label, penwidth=get_penwidth(prob_N1_A_label))
    dot.edge("N1", "N1_Q", prob_N1_Q_label, penwidth=get_penwidth(prob_N1_Q_label))

    # After N=0
    dot.node("N0_A", "Audit")
    dot.node("N0_Q", "Query M")
    prob_N0_A_label = f"p(A|N=0): {best_x[6]:.4f}"
    prob_N0_Q_label = f"p(Q|N=0): {best_x[8]:.4f}"
    dot.edge("N0", "N0_A", prob_N0_A_label, penwidth=get_penwidth(prob_N0_A_label))
    dot.edge("N0", "N0_Q", prob_N0_Q_label, penwidth=get_penwidth(prob_N0_Q_label))

    # Fourth level: Second monitor signals and final decisions
    # After M=1 -> Query N
    dot.node("M1N1", "N=1")
    dot.node("M1N0", "N=0")
    prob_M1N1_A_label = f"p(A|M=1,N=1): {best_x[9]:.4f}"
    prob_M1N0_A_label = f"p(A|M=1,N=0): {best_x[10]:.4f}"
    dot.edge("M1_Q", "M1N1", penwidth="1")  # Default width for non-probability edges
    dot.edge("M1_Q", "M1N0", penwidth="1")  # Default width for non-probability edges
    dot.edge(
        "M1N1", "Audit", prob_M1N1_A_label, penwidth=get_penwidth(prob_M1N1_A_label)
    )
    dot.edge(
        "M1N0", "Audit", prob_M1N0_A_label, penwidth=get_penwidth(prob_M1N0_A_label)
    )

    # After M=0 -> Query N
    dot.node("M0N1", "N=1")
    dot.node("M0N0", "N=0")
    prob_M0N1_A_label = f"p(A|M=0,N=1): {best_x[11]:.4f}"
    prob_M0N0_A_label = f"p(A|M=0,N=0): {best_x[12]:.4f}"
    dot.edge("M0_Q", "M0N1", penwidth="1")  # Default width for non-probability edges
    dot.edge("M0_Q", "M0N0", penwidth="1")  # Default width for non-probability edges
    dot.edge(
        "M0N1", "Audit", prob_M0N1_A_label, penwidth=get_penwidth(prob_M0N1_A_label)
    )
    dot.edge(
        "M0N0", "Audit", prob_M0N0_A_label, penwidth=get_penwidth(prob_M0N0_A_label)
    )

    # After N=1 -> Query M
    dot.node("N1M1", "M=1")
    dot.node("N1M0", "M=0")
    prob_N1M1_A_label = f"p(A|N=1,M=1): {best_x[9]:.4f}"
    prob_N1M0_A_label = f"p(A|N=1,M=0): {best_x[11]:.4f}"
    dot.edge("N1_Q", "N1M1", penwidth="1")  # Default width for non-probability edges
    dot.edge("N1_Q", "N1M0", penwidth="1")  # Default width for non-probability edges
    dot.edge(
        "N1M1", "Audit", prob_N1M1_A_label, penwidth=get_penwidth(prob_N1M1_A_label)
    )
    dot.edge(
        "N1M0", "Audit", prob_N1M0_A_label, penwidth=get_penwidth(prob_N1M0_A_label)
    )

    # After N=0 -> Query M
    dot.node("N0M1", "M=1")
    dot.node("N0M0", "M=0")
    prob_N0M1_A_label = f"p(A|N=0,M=1): {best_x[10]:.4f}"
    prob_N0M0_A_label = f"p(A|N=0,M=0): {best_x[12]:.4f}"
    dot.edge("N0_Q", "N0M1", penwidth="1")  # Default width for non-probability edges
    dot.edge("N0_Q", "N0M0", penwidth="1")  # Default width for non-probability edges
    dot.edge(
        "N0M1", "Audit", prob_N0M1_A_label, penwidth=get_penwidth(prob_N0M1_A_label)
    )
    dot.edge(
        "N0M0", "Audit", prob_N0M0_A_label, penwidth=get_penwidth(prob_N0M0_A_label)
    )

    return dot


def update_from_slider(base_key, slider_key):
    """
    Callback to update the shared variable from the slider's value.
    When the slider changes, we set st.session_state[base_key] equal to the slider's value.
    """
    st.session_state[base_key] = st.session_state[slider_key]


def update_from_num(base_key, num_key):
    """
    Callback to update the shared variable from the number input's value.
    When the number input changes, we set st.session_state[base_key] equal to the number input's value.
    """
    st.session_state[base_key] = st.session_state[num_key]


def main():
    # Initialize session state for sliders if not done yet
    if "initialized_sliders" not in st.session_state:
        for i in range(13):
            st.session_state[f"x{i}"] = 0.0
        st.session_state["p_a_q1_current"] = (
            None  # Will store P(A|Q=1) from current sliders
        )
        st.session_state["p_a_q0_current"] = (
            None  # Will store P(A|Q=0) from current sliders
        )
        st.session_state["e_cost_q0_current"] = None
        st.session_state["best_obj"] = None  # Best P(A|Q=1) from the solver
        st.session_state["initialized_sliders"] = True
        st.session_state["use_extra_audit_budget"] = False

    if "metadata" not in st.session_state:
        st.session_state["metadata"] = {}

    st.title("Multiple Monitor Optimization with One Unified Budget")
    with st.expander("Quick summary of the model:"):
        st.markdown(
            """
# AI Output Monitoring System

See draft of paper report here: https://drive.google.com/file/d/16Zq5LFiCpwrlgUUe2UX3XMkKdIcvAoK4/view?usp=sharing
                    
## Overview
The system aims to detect whether an AI model's output ($Q$) is aligned ($Q=0$) or maligned ($Q=1$) using two monitors (M and N). A unified budget constraint applies to both monitoring and auditing costs.

## Key Components
* **Monitors**: Two binary monitors ($M, N$) that provide signals, each with associated costs ($c_m, c_n$)
* **Auditing**: If we audit the output, we'll have to pay $c_a$
* **Budget Constraint**: Total expected cost when $Q=0$ must equal budget $Y$

## Decision Process
After receiving a signal from the first monitor, three options are available:
* Audit immediately
* Query the second monitor
* Conclude without further action

## Strategy Tree Visualization
The provided graph shows a complete decision tree where:
* Initial choice between monitors M and N
* Each edge shows conditional probabilities for next actions
* Terminal nodes all lead to either "Audit" or an implicit "Conclude"

## Optimization Goal
Maximize $P(A=1|Q=1)$ subject to the unified budget constraint.
        """
        )

    left_col, middle_col, right_col = st.columns([1, 1, 1.5])

    ###############################################################################
    # LEFT COLUMN
    ###############################################################################
    with left_col:
        # Place the "Compute Optimal Strategy" button at the top
        if st.button("Compute Optimal Strategy"):
            try:

                # Parse the JSON input
                prob = json.loads(st.session_state["prob_json_text"])

                # Solve with the existing function solve_with_sqp from above (already defined)
                best_x, best_obj_val, solver_metadata = solve_with_sqp(
                    prob,
                    st.session_state["c_m"],
                    st.session_state["c_n"],
                    st.session_state["c_a"],
                    st.session_state["Y"],
                    st.session_state["T"],
                    n_starts=20,
                    use_extra_audit_budget=st.session_state["use_extra_audit_budget"],
                )
                st.session_state["metadata"] = solver_metadata
                if best_x is not None:
                    # Update session-state sliders with the solver result
                    for i in range(13):
                        st.session_state[f"x{i}"] = float(best_x[i])
                    st.session_state["best_obj"] = float(best_obj_val)
                else:
                    st.error("No feasible solution found under these constraints.")

            except json.JSONDecodeError:
                st.error("Invalid JSON format in probability parameters.")
            except Exception as e:
                st.error(f"Error during calculation: {str(e)}")

        st.subheader("Input Parameters")

        default_probs = {
            "p_m_given_q": 0.7916666666666666,
            "p_m_given_not_q": 0.08508771929824561,
            "p_n_given_q": 0.7416666666666667,
            "p_n_given_not_q": 0.18070175438596492,
            "p_m1n0_given_q": 0.125,
            "p_m0n1_given_q": 0.075,
            "p_m0n0_given_q": 0.13333333333333333,
            "p_m1n1_given_q": 0.6666666666666667,
            "p_m1n0_given_not_q": 0.039473684210526314,
            "p_m0n1_given_not_q": 0.13508771929824562,
            "p_m0n0_given_not_q": 0.7798245614035088,
            "p_m1n1_given_not_q": 0.04561403508771933,
        }

        # Use or initialize session state for the JSON text
        if "prob_json_text" not in st.session_state:
            st.session_state["prob_json_text"] = json.dumps(default_probs, indent=2)

        st.session_state["prob_json_text"] = st.text_area(
            "Enter probability parameters (JSON format):",
            value=st.session_state["prob_json_text"],
            height=300,
        )

        st.write("Here are our budget constraints and costs")

        if "T" not in st.session_state:
            st.session_state["T"] = 0.10
        if "Y" not in st.session_state:
            st.session_state["Y"] = 2.0
        if "c_m" not in st.session_state:
            st.session_state["c_m"] = 1.0
        if "c_n" not in st.session_state:
            st.session_state["c_n"] = 0.18
        if "c_a" not in st.session_state:
            st.session_state["c_a"] = 20.0
        if "use_extra_audit_budget" not in st.session_state:
            st.session_state["use_extra_audit_budget"] = False

        st.session_state["Y"] = st.number_input(
            "Total Budget Constraint (Y) (If using audit-specific budget, this is the monitor budget)",
            value=st.session_state["Y"],
            min_value=0.0,
        )
        st.session_state["c_m"] = st.number_input(
            "Cost of Monitor M (c_m)", value=st.session_state["c_m"], min_value=0.0
        )
        st.session_state["c_n"] = st.number_input(
            "Cost of Monitor N (c_n)", value=st.session_state["c_n"], min_value=0.0
        )
        st.session_state["c_a"] = st.number_input(
            "Cost of Audit (c_a)", value=st.session_state["c_a"], min_value=0.0
        )
        st.session_state["use_extra_audit_budget"] = st.checkbox(
            "Use Audit-Specific Budget",
            value=st.session_state["use_extra_audit_budget"],
        )

        if st.session_state["use_extra_audit_budget"]:
            st.session_state["T"] = st.number_input(
                "Audit-Specific Budget Constraint (T)",
                value=st.session_state["T"],
                min_value=0.0,
                max_value=1.0,
            )

        prob = json.loads(st.session_state["prob_json_text"])

    ###############################################################################
    # MIDDLE COLUMN
    ###############################################################################
    with middle_col:
        # Put the "Refresh" button at the top
        if st.button("Refresh"):
            try:
                prob = json.loads(st.session_state["prob_json_text"])
                # Make sure needed functions from above are in scope:
                x_current = [st.session_state[f"x{i}"] for i in range(13)]
                p_a_q1 = p_a_given_q1(x_current, prob)
                p_a_q0 = p_a_given_q0(x_current, prob)
                e_cost_q0 = expected_monitor_cost_q0(
                    x_current, prob, st.session_state["c_m"], st.session_state["c_n"]
                )
                st.session_state["p_a_q1_current"] = p_a_q1
                st.session_state["p_a_q0_current"] = p_a_q0
                st.session_state["e_cost_q0_current"] = e_cost_q0

            except json.JSONDecodeError:
                st.error("Invalid JSON format in probability parameters.")
            except Exception as e:
                st.error(f"Error during recalculation: {str(e)}")

        st.subheader("Monitoring Strategy Inputs")

        # Define the labels for the 13 strategy variables.
        var_labels = [
            "p(M)",
            "p(A|M=1)",
            "p(A|M=0)",
            "p(Q|M=1)",
            "p(Q|M=0)",
            "p(A|N=1)",
            "p(A|N=0)",
            "p(Q|N=1)",
            "p(Q|N=0)",
            "p(A|M=1,N=1)",
            "p(A|M=1,N=0)",
            "p(A|M=0,N=1)",
            "p(A|M=0,N=0)",
        ]

        x_current = []
        for i in range(13):
            base_key = f"x{i}"
            # Create two columns with 70/30 split
            col1, col2 = st.columns([0.7, 0.3])

            with col1:
                # Slider widget with unique key and a callback to update the shared variable
                st.slider(
                    f"{var_labels[i]}",
                    0.0,
                    1.0,
                    value=float(st.session_state[base_key]),
                    step=0.0001,
                    key=f"{base_key}_slider",
                    on_change=lambda base_key=base_key, slider_key=f"{base_key}_slider": update_from_slider(
                        base_key, slider_key
                    ),
                )

            with col2:
                # Number input widget with unique key and its own callback
                st.number_input(
                    "",  # Empty label since it's next to slider
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state[base_key]),
                    step=0.0001,
                    format="%.4f",  # Match number of decimal places with step
                    key=f"{base_key}_num",
                    on_change=lambda base_key=base_key, num_key=f"{base_key}_num": update_from_num(
                        base_key, num_key
                    ),
                )

            # After both widgets, we read the current synchronized value
            x_current.append(st.session_state[base_key])

        st.write(f"Raw slider values synced with number inputs: {x_current}")

    ###############################################################################
    # RIGHT COLUMN (RESULTS)
    ###############################################################################
    with right_col:
        st.subheader("Results")

        # Put the solved maximum caught rate at the top
        if st.session_state["best_obj"] is not None:
            st.markdown(
                f"**Best Solved Catch Rate P(A=1|Q=1): {st.session_state['best_obj']:.4f}**"
            )
        else:
            st.write("No solution found. Click 'Compute Optimal Strategy' to find one.")
        if st.session_state["metadata"].get("unstable_solve", False):
            st.write("Warning: The solver did not converge to a stable solution.")
            st.write(
                f"Top 5 solutions: {st.session_state['metadata']['best_vals'][:5]}"
            )
            st.write(
                f"Top 5 solutions parameters: {st.session_state['metadata']['solved_parameters'][:5]}"
            )

        # Current slider-based results (if computed)
        if st.session_state["p_a_q1_current"] is not None:
            st.write(
                f"Current Slider-Based Catch Rate P(A=1|Q=1): {st.session_state['p_a_q1_current']:.4f}"
            )
            st.write(
                f"""Current Slider-Based total cost: {combined_monitor_and_audit_cost(
                    x_current, prob, st.session_state["c_m"], st.session_state["c_n"], st.session_state["c_a"]
                    ):.4f}"""
            )
            st.write(
                f"Current Slider-Based Audit False Positive Rate P(A=1|Q=0): {st.session_state['p_a_q0_current']:.4f}"
            )
            st.write(
                f"Current Slider-Based Monitoring Specific Cost E[C(A)|Q=0]: {st.session_state['e_cost_q0_current']:.4f}"
            )

            st.markdown("### Graph of the monitoring game tree")

        # If we have a current strategy, show the game tree
        if st.session_state["p_a_q1_current"] is not None:
            x_sliders = [st.session_state[f"x{i}"] for i in range(13)]
            game_tree = create_game_tree(x_sliders)
            st.graphviz_chart(game_tree)
        else:
            st.write(
                "No game tree available. Click 'Refresh' to generate one with your current sliders."
            )


if __name__ == "__main__":
    main()

# BUG with Y = 2, c_m =1, c_n = 0.6, c_a = 20.
# Might want to up the number of starts and report like the variance among the top ten.

# x = [0.30434782608709166, 0.9999999999999993, 4.440892098500626e-16, 1.6653345369377348e-16, 0.0, 0.0, 0.0, 1.0, 0.0, 0.9999999999999994, 1.0, 0.0, 0.030969701044365923]


# Potential bug with audit constrain 0.1, Y = 1.1, c_m = 1. c_n = 0.6. We get the case where we still query N when M = 1, but always audit when we do,
# so it's like a waste of a query.


# Oh I think here the cost constraint is actually just straight up slack and it's the audit constraint that bounds!
