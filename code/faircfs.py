import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
import subprocess

def is_independent(X, Y, given_set, data, threshold=0.01, n_bins=5):
    """
    General conditional independence test using conditional mutual information (CMI).

    Args:
        X (str): Feature name.
        Y (str): Feature name.
        given_set (set): Conditioning set of feature names.
        data (pd.DataFrame): Dataset containing all variables.
        threshold (float): Threshold below which we consider independence.
        n_bins (int): Number of bins for discretization.

    Returns:
        bool: True if X ⟂ Y | given_set, else False.
    """

    # Extract variables
    x = data[X].values.reshape(-1, 1)
    y = data[Y].values.reshape(-1, 1)

    # If no conditioning, test plain MI
    if not given_set:
        contingency = np.histogram2d(x.ravel(), y.ravel(), bins=n_bins)[0]
        mi = mutual_info_score(None, None, contingency=contingency)
        return mi < threshold

    # Condition on given_set by regressing out Z
    Z = data[list(given_set)].values

    def residuals(a, Z):
        """Return residuals of linear regression of a on Z."""
        model = LinearRegression().fit(Z, a)
        return a - model.predict(Z)

    x_res = residuals(x, Z)
    y_res = residuals(y, Z)

    # Compute MI between residuals
    contingency = np.histogram2d(x_res.ravel(), y_res.ravel(), bins=n_bins)[0]
    mi = mutual_info_score(None, None, contingency=contingency)

    return mi < threshold


def fair_cfs(S):
    """
    Optimized FairCFS algorithm with forward-selection for Step 3.

    Args:
        S (str): Sensitive feature.
    Returns:
        set: Fair causal features (T).
    """
    data = pd.read_csv("data.csv")
    # Step 1: Discover MBs
    MB_Y = set(get_mb("target"))
    MB_S = set(get_mb("gender"))

    M1, M2 = set(), set()

    # Step 2: Features in MB(Y) \ MB(S)
    for X in MB_Y - MB_S:
        if is_independent(X, S, MB_S, data):
            M1.add(X)

    # Step 3: Features in MB(Y) ∩ MB(S)
    for X in MB_Y & MB_S:
        remaining = list(MB_S - {X})
        Z = set()

        # Forward-selection heuristic
        improved = True
        while improved and not is_independent(X, S, Z, data):
            improved = False
            for candidate in remaining:
                test_set = Z | {candidate}
                if is_independent(X, S, test_set, data):
                    Z = test_set
                    improved = True
                    break  # accept first working candidate

        if is_independent(X, S, Z, data):
            M2.add(X)

    return M1 | M2

def get_mb(mb_var):
    if mb_var == "target":
        result = subprocess.run(
            ['Rscript', 'bnlearn_hc_disc.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )
    else:
        result = subprocess.run(
            ['Rscript', 'bnlearn_hc_gender_mb.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )

    output = result.stdout.strip()

    column_names = output.split()
    return column_names
