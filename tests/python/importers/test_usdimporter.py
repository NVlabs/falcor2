# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import pytest
import slangpy as spy
import falcor2 as f2

DATA = Path(__file__).parent.parent.parent.parent / "data"


def test_load_scene():
    usd_importer = f2.UsdImporter()
    scene = usd_importer.load_scene(DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda")
    assert scene is not None

    for mesh in scene.meshes:
        assert mesh.positions is not None
        assert mesh.normals is not None
        assert mesh.tangents is not None
        assert mesh.handedness is not None
        tc0 = mesh.texcoords(0)
        assert tc0 is not None
        assert mesh.positions.shape[0] == mesh.vertex_count
        assert mesh.normals.shape[0] == mesh.vertex_count
        assert mesh.tangents.shape[0] == mesh.vertex_count
        assert mesh.handedness.shape[0] == mesh.vertex_count
        assert tc0.shape[0] == mesh.vertex_count
        # Note: Individual vertex access is no longer available in the new stream-based API
        # The above assertions verify that all attribute arrays have the correct length

        for subgeo in mesh.subgeometries:
            indices1D = subgeo.indices_numpy.flatten()
            for i, tri in enumerate(subgeo.indices):
                assert (subgeo.indices_numpy[i] == tri).all()
                assert spy.math.all(spy.bool3([indices1D[i * 3 + j] for j in range(3)] == tri))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
