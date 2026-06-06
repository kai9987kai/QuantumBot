import importlib.util

import numpy as np
import pytest

from quantumbot.quantum import (
    N_FEATURES,
    PARAMETER_COUNT,
    VariationalQuantumPolicy,
)


def test_policy_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(42)
    policy = VariationalQuantumPolicy.random(rng)
    features = np.linspace(-1.0, 1.0, N_FEATURES)

    first = policy.act(features)
    second = policy.act(features)

    np.testing.assert_allclose(first, second)
    assert first.shape == (2,)
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)


def test_policy_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="parameters"):
        VariationalQuantumPolicy(np.zeros(PARAMETER_COUNT - 1))

    policy = VariationalQuantumPolicy(np.zeros(PARAMETER_COUNT))
    with pytest.raises(ValueError, match="features"):
        policy.act(np.zeros(N_FEATURES - 1))


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None,
    reason="Qiskit is an optional dependency",
)
def test_numpy_backend_matches_qiskit() -> None:
    rng = np.random.default_rng(123)
    parameters = rng.normal(size=PARAMETER_COUNT)
    features = rng.uniform(-1.0, 1.0, size=N_FEATURES)

    numpy_result = VariationalQuantumPolicy(parameters, "numpy").act(features)
    qiskit_result = VariationalQuantumPolicy(parameters, "qiskit").act(features)

    np.testing.assert_allclose(numpy_result, qiskit_result, atol=1e-10)
