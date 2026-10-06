import numpy as np

from molecular_compiler import Stimulus, compile, simulate
from molecular_compiler.body import BodyModel, MappedBodyAdapter, SpringBody


def test_M13_R1_closed_loop_boundary_and_mapping(system):
    sim = compile(*system)
    body = MappedBodyAdapter(
        SpringBody(),
        4,
        (0,),
        (3,),
        backend_provenance={"kind": "synthetic", "version": "1"},
    )
    assert isinstance(body, BodyModel)
    trajectory = simulate(sim, Stimulus(), 0.003, body=body)
    assert trajectory.metadata["boundary_condition"] == "closed_loop"
    assert np.all(np.isfinite(trajectory.voltage))
    assert body.backend.position != 0
