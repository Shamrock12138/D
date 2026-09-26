from pathlib import Path

from transport_map_base import (
    CODE_ROOT,
    TransportMapBase,
)


class P005TransportMap(
    TransportMapBase
):
    """
    P005 运输路线与多点访问结构。

    P005:
        25 个架次
        20 个单站
        5 个双站
    """

    def __init__(self):

        schedule_file = (
            CODE_ROOT
            / "data"
            / "q2_compact_moead"
            / "P005"
            / "schedule.csv"
        )

        super().__init__(
            schedule_file=schedule_file,
            output_stem=(
                "q2_p005_transport_routes"
            ),
            expected_single=20,
            expected_double=5,
            figsize=(7.5, 6.6),
        )


def main():
    figure = (
        P005TransportMap()
    )

    figure.draw()


if __name__ == "__main__":
    main()