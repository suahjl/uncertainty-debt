# %%
import numpy as np
import matplotlib.pyplot as plt

# Modified Parameters for Negative Response
beta = 0.99  # Discount factor
sigma = 1.5  # Increased intertemporal elasticity of substitution
kappa = 0.3  # Increased slope of Phillips curve
phi_pi = 1.5  # Decreased response to inflation
phi_y = 0.5  # Taylor rule: reaction to output gap
r_n = 0.02  # Natural interest rate
rho = 0.1  # Sensitivity of the spread to the output gap
theta = 0.75  # Calvo pricing parameter (probability of not changing prices)
lambda_w = 0.75  # Increased downward nominal wage rigidity
debt_threshold = 0.15  # Debt threshold

T = 16  # Time periods for IRF


# Initial values for different cases
def simulate_model(average_debt, shock_type):
    y = np.zeros(T)
    pi = np.zeros(T)
    r = np.zeros(T)
    rb = np.zeros(T)  # Borrowing rate
    default_risk = np.zeros(T)  # Default risk measure
    uncertainty_shock = np.zeros(T)

    # Debt levels
    household_debt = np.full(T, average_debt)

    # Impulse shocks
    monetary_shock = np.zeros(T)
    uncertainty_shock = np.zeros(T)

    if shock_type == "monetary":
        monetary_shock[0] = 1  # % monetary policy shock at time 0
    elif shock_type == "uncertainty":
        uncertainty_shock[0] = 1  # % uncertainty shock at time 0

    # Model simulation with all features
    for t in range(1, T):
        # Expectations with uncertainty shock affecting expectations
        if uncertainty_shock[t - 1] > 0:
            E_pi_next = pi[t] * (
                1 + uncertainty_shock[t - 1]
            )  # Bounded expectation due to uncertainty shock
        else:
            E_pi_next = pi[t]  # Unbounded expectation  (or t-1 for backward RE)

        E_y_next = y[t - 1]

        # Default risk calculation
        default_risk[t] = np.maximum(0, household_debt[t] - debt_threshold)

        # Spread as a function of default risk and output gap
        spread = rho * y[t - 1] + default_risk[t]

        # Borrowing rate (including spread) --- NOTE: MAYBE SPLIT THIS INTO TWO 
        rb[t] = r[t - 1] + spread

        # IS Curve with borrowing rate
        y[t] = E_y_next - sigma * (rb[t] - E_pi_next - r_n)

        # Phillips Curve with Calvo pricing
        pi[t] = (beta * E_pi_next + kappa * y[t]) * (1 - theta) + theta * pi[t - 1]

        # Wage rigidity affecting the output gap 
        y[t] = np.maximum(y[t], -lambda_w)

        # Taylor Rule (with both monetary and uncertainty shocks) --- NOTE: SHOULD ONLY HAVE ONE RATE
        r[t] = r_n + phi_pi * pi[t] + phi_y * y[t] + monetary_shock[t]

    return y, pi, r, rb, default_risk, uncertainty_shock


# Function to plot IRFs for both below and above threshold
def plot_irfs_below_above(
    y_below,
    pi_below,
    r_below,
    rb_below,
    default_risk_below,
    y_above,
    pi_above,
    r_above,
    rb_above,
    default_risk_above,
    uncertainty_shock,
    shock_type,
):

    plt.figure(figsize=(12, 10))

    plt.subplot(6, 1, 1)
    plt.plot(y_below, label="Output Gap (y_t) - Below Threshold")
    plt.plot(y_above, label="Output Gap (y_t) - Above Threshold", linestyle="--")
    plt.title(f"IRFs with Financial Frictions and {shock_type.capitalize()} Shock")
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(pi_below, label="Inflation (π_t) - Below Threshold")
    plt.plot(pi_above, label="Inflation (π_t) - Above Threshold", linestyle="--")
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(r_below, label="Interest Rate (r_t) - Below Threshold")
    plt.plot(r_above, label="Interest Rate (r_t) - Above Threshold", linestyle="--")
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.plot(rb_below, label="Borrowing Rate (r_b_t) - Below Threshold")
    plt.plot(rb_above, label="Borrowing Rate (r_b_t) - Above Threshold", linestyle="--")
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.plot(default_risk_below, label="Default Risk - Below Threshold")
    plt.plot(default_risk_above, label="Default Risk - Above Threshold", linestyle="--")
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(uncertainty_shock, label=f"{shock_type.capitalize()} Shock")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Simulate for below threshold 
average_debt_below = 0.1
average_debt_above = 0.5

# Simulate IRFs for a monetary shock
y_below, pi_below, r_below, rb_below, default_risk_below, uncertainty_shock = (
    simulate_model(average_debt_below, "monetary")
)
y_above, pi_above, r_above, rb_above, default_risk_above, _ = simulate_model(
    average_debt_above, "monetary"
)

plot_irfs_below_above(
    y_below,
    pi_below,
    r_below,
    rb_below,
    default_risk_below,
    y_above,
    pi_above,
    r_above,
    rb_above,
    default_risk_above,
    uncertainty_shock,
    "monetary",
)

# Simulate IRFs for an uncertainty shock
y_below, pi_below, r_below, rb_below, default_risk_below, uncertainty_shock = (
    simulate_model(average_debt_below, "uncertainty")
)
y_above, pi_above, r_above, rb_above, default_risk_above, _ = simulate_model(
    average_debt_above, "uncertainty"
)

plot_irfs_below_above(
    y_below,
    pi_below,
    r_below,
    rb_below,
    default_risk_below,
    y_above,
    pi_above,
    r_above,
    rb_above,
    default_risk_above,
    uncertainty_shock,
    "uncertainty",
)

# %%
