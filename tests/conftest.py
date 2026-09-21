"""Small independent fixtures with transparent hand-computable profiles."""

import pytest

from geo_llm_scheduler.domain.models import (
    Job,
    ProblemInstance,
    Profile,
    Region,
    ServingInstance,
    Tariff,
)


@pytest.fixture
def problem():
    return ProblemInstance(
        (
            Job("a", 0, Profile(100, 0.5, 2), Profile(200, 0.5, 2), 30),
            Job("b", 0, Profile(100, 0.5, 2), Profile(200, 0.5, 2), 30),
        ),
        (Region("r0", (Tariff(0, 10000, 0.2),), 2), Region("r1", (Tariff(0, 10000, 0.1),), 3)),
        (
            ServingInstance("m0", 0, 8, 10, 100),
            ServingInstance("m1", 0, 8, 10, 100),
            ServingInstance("m2", 1, 8, 10, 100),
        ),
    )
