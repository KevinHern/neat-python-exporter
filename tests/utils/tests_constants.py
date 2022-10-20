from neat_python_utility.tests.utils.test_case_model import TestCaseModel


class TestCases:
    SIMPLE_NN = TestCaseModel(
        id_inputs={-1, -2, -3}, id_outputs={4},
        connections_keys=[
            (-1, 1), (-2, 2), (-3, 3),
            (1, 4), (2, 4), (3, 4),
        ],
        nodes_indexes=[1, 2, 3, 4]
    )

    MULTIPLE_OUTPUTS_NN = TestCaseModel(
        id_inputs={-1, -2, -3}, id_outputs={4, 5, 6},
        connections_keys=[
            (-1, 1), (-2, 2), (-3, 3),
            (1, 4), (2, 4), (3, 4), (1, 5), (3, 5), (2, 6),
        ],
        nodes_indexes=[1, 2, 3, 4, 5, 6]
    )

    TWO_HIDDEN_LAYERS_NN = TestCaseModel(
        id_inputs={-1, -2, -3, -4}, id_outputs={5},
        connections_keys=[
            (-1, 1), (-2, 2), (-3, 3), (-4, 4),
            (1, 6), (2, 6), (3, 7), (4, 7), (1, 8), (4, 8),
            (6, 9), (7, 9), (7, 10), (8, 10),
            (9, 5), (10, 5)
        ],
        nodes_indexes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    SKIPPED_INPUT_TO_OUTPUT_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-1, 15), (-2, 15),
            (15, 0), (-1, 0), (-2, 0)
        ],
        nodes_indexes=[15, 0]
    )

    ABANDONED_NODES_ONE_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-1, 41), (-2, 41),
            (41, 1073),
            (-1, 0), (41, 0)
        ],
        nodes_indexes=[41, 1073, 0]
    )

    ABANDONED_NODES_TWO_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={107},
        connections_keys=[
            (-1, 0), (-2, 0), (-1, 378),
            (0, 421),
            (421, 107), (0, 107), (-1, 107), (552, 107)
        ],
        nodes_indexes=[0, 378, 421, 552, 107]
    )

    ABANDONED_NODES_THREE_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-1, 291), (-2, 291),
            (291, 0),
            (-1, 555), (-2, 555)
        ],
        nodes_indexes=[291, 0]
    )

    WEIRD_TOPOLOGY_ONE_NN = TestCaseModel(
        id_inputs={-1, -2, -3}, id_outputs={4},
        connections_keys=[
            (-1, 1), (-2, 2), (-3, 3),
            (2, 5), (3, 5),
            (1, 6), (5, 6),
            (1, 4), (6, 4), (5, 4), (3, 4)
        ],
        nodes_indexes=[1, 2, 3, 4, 5, 6]
    )

    WEIRD_TOPOLOGY_TWO_NN = TestCaseModel(
        id_inputs={-1, -2, -3}, id_outputs={4, 5},
        connections_keys=[
            (-1, 1), (-2, 2), (-3, 3),
            (1, 6), (2, 6), (3, 6),
            (1, 7), (6, 7),
            (1, 4), (3, 4), (7, 4), (2, 5), (6, 5)
        ],
        nodes_indexes=[1, 2, 3, 4, 5, 6, 7]
    )

    WEIRD_TOPOLOGY_THREE_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-1, 101), (-2, 101),
            (101, 0), (101, 1751), (-2, 1751),
            (-2, 1718), (1751, 1718),
            (-2, 348), (1718, 348)
        ],
        nodes_indexes=[101, 0, 1751, 1718, 348]
    )

    WEIRD_TOPOLOGY_FOUR_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-1, 451), (-2, 451),
            (-1, 0), (-2, 0), (451, 0),
            (0, 1319),
            (-1, 1457), (1583, 1457),
            (1457, 1609),
            (451, 967), (-1, 967), (1457, 967), (1609, 967)
        ],
        nodes_indexes=[451, 0, 1319, 1457, 1609, 967, 1583]
    )

    ASYMMETRIC_TOPOLOGY_ONE_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-2, 1024),
            (1024, 1666),
            (-1, 51), (-2, 51), (1024, 51), (1666, 51),
            (51, 0),
            (-2, 808)
        ],
        nodes_indexes=[1024, 1666, 51, 0]
    )

    ASYMMETRIC_TOPOLOGY_TWO_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-1, 320),
            (-1, 398), (-2, 398), (320, 398),
            (-2, 513), (398, 513),
            (320, 0), (513, 0),
        ],
        nodes_indexes=[320, 398, 513, 0]
    )

    ASYMMETRIC_TOPOLOGY_THREE_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-2, 865),
            (-1, 1850), (865, 1850),
            (-1, 168), (-2, 168), (865, 168), (1850, 168),
            (168, 0),
        ],
        nodes_indexes=[865, 1850, 168, 0, 130]
    )

    ASYMMETRIC_TOPOLOGY_FOUR_NN = TestCaseModel(
        id_inputs={-1, -2}, id_outputs={0},
        connections_keys=[
            (-1, 897), (-1, 967),
            (-2, 49), (897, 49), (967, 49),
            (49, 0),
        ],
        nodes_indexes=[897, 967, 49, 0]
    )
