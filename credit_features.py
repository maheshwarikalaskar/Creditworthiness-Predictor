"""Feature engineering shared by the training notebook and the Streamlit app.

It lives in its own module so that the pickled model can be loaded anywhere this file is importable.
"""


def add_credit_per_month(X):
    """Add the engineered feature: monthly instalment size (credit amount spread over the duration)."""
    X = X.copy()
    X["credit_per_month"] = X["credit_amount"] / (X["duration"] + 1)
    return X
