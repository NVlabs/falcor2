# SPDX-License-Identifier: Apache-2.0

from falcor2.utils.udim_utils import decode_udim_index


def test_decode_udim_index():
    assert decode_udim_index(1001) == (0, 0)
    assert decode_udim_index(1002) == (1, 0)
    assert decode_udim_index(1011) == (0, 1)
    assert decode_udim_index(1015) == (4, 1)
    assert decode_udim_index(1025) == (4, 2)
    assert decode_udim_index(1034) == (3, 3)
