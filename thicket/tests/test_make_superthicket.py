# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from thicket import Thicket as th


def test_make_superthicket(mpi_scaling_cali):
    th_list = []
    for file in mpi_scaling_cali:
        th_list.append(th.from_caliperreader(file))

    # Add arbitrary value to statsframe
    t_val = 0
    for t in th_list:
        t.statsframe.dataframe["test"] = t_val
        t_val += 2

    superthicket = th.make_superthicket(th_list)

    # Check level values
    assert set(superthicket.dataframe.index.get_level_values("thicket")) == {
        0,
        1,
        2,
        3,
        4,
    }
    # Check PerfData values
    assert set(superthicket.dataframe["test"]) == {0, 2, 4, 6, 8}

    superthicket_named = th.make_superthicket(
        th_list, profiles_from_meta="mpi.world.size"
    )

    # Check level values
    assert set(superthicket_named.dataframe.index.get_level_values("thicket")) == {
        27,
        64,
        125,
        216,
        343,
    }
    # Check PerfData values
    assert set(superthicket_named.dataframe["test"]) == {0, 2, 4, 6, 8}
