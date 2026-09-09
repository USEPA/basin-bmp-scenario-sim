from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.bmp import _sample_efficiency_map
from src.constants import CFG_BMP_EFFICIENCY, PATHWAY_VALUES
from src.input_config import _load_bmp_efficiency


class RecordingLogger:
    def __init__(self) -> None:
        self.verbose_messages: list[str] = []
        self.warning_messages: list[str] = []

    def verbose(self, message, *args, **kwargs) -> None:
        self.verbose_messages.append(str(message))

    def warning(self, message, *args, **kwargs) -> None:
        self.warning_messages.append(str(message))


def test_missing_subsurface_efficiencies_are_completed_with_logged_zeros(tmp_path) -> None:
    efficiency_path = tmp_path / "bmp_efficiency.csv"
    pd.DataFrame(
        [
            {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.25},
            {
                "cps": 340,
                "pollutant": "TN",
                "pathway": "shallow subsurface",
                "value": 0.10,
            },
            {"cps": 340, "pollutant": "TP", "pathway": "surface", "value": 0.40},
        ]
    ).to_csv(efficiency_path, index=False)
    logger = RecordingLogger()

    loaded = _load_bmp_efficiency(
        {CFG_BMP_EFFICIENCY: str(efficiency_path)},
        cps=[340],
        pollutants=["TN", "TP"],
        logger=logger,
    )

    assert set(
        loaded[["cps", "pollutant", "pathway"]].itertuples(index=False, name=None)
    ) == {
        (340, pollutant, pathway)
        for pollutant in ("TN", "TP")
        for pathway in PATHWAY_VALUES
    }
    values = loaded.set_index(["cps", "pollutant", "pathway"])["value"]
    assert values.loc[(340, "TN", "surface")] == pytest.approx(0.25)
    assert values.loc[(340, "TN", "shallow subsurface")] == pytest.approx(0.10)
    assert values.loc[(340, "TN", "deep subsurface")] == pytest.approx(0.0)
    assert values.loc[(340, "TP", "shallow subsurface")] == pytest.approx(0.0)
    assert values.loc[(340, "TP", "deep subsurface")] == pytest.approx(0.0)

    default_messages = [
        message
        for message in logger.verbose_messages
        if "assuming efficiency=0" in message
    ]
    assert len(default_messages) == 3
    assert any("pollutant=TN" in message and "deep subsurface" in message for message in default_messages)
    assert any("pollutant=TP" in message and "shallow subsurface" in message for message in default_messages)
    assert any("pollutant=TP" in message and "deep subsurface" in message for message in default_messages)


def test_legacy_efficiency_rows_are_surface_only_and_subsurface_defaults_to_zero(tmp_path) -> None:
    efficiency_path = tmp_path / "bmp_efficiency.csv"
    pd.DataFrame(
        [{"cps": 340, "pollutant": "TN", "value": 0.35}]
    ).to_csv(efficiency_path, index=False)
    logger = RecordingLogger()

    loaded = _load_bmp_efficiency(
        {CFG_BMP_EFFICIENCY: str(efficiency_path)},
        cps=[340],
        pollutants=["TN"],
        logger=logger,
    )

    values = loaded.set_index("pathway")["value"].to_dict()
    assert values == {
        "surface": pytest.approx(0.35),
        "shallow subsurface": pytest.approx(0.0),
        "deep subsurface": pytest.approx(0.0),
    }
    assert any("has no pathway column" in message for message in logger.verbose_messages)
    assert sum("assuming efficiency=0" in message for message in logger.verbose_messages) == 2


def test_blank_subsurface_efficiency_is_defaulted_to_zero(tmp_path) -> None:
    efficiency_path = tmp_path / "bmp_efficiency.csv"
    pd.DataFrame(
        [
            {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.25},
            {
                "cps": 340,
                "pollutant": "TN",
                "pathway": "shallow subsurface",
                "value": None,
            },
        ]
    ).to_csv(efficiency_path, index=False)
    logger = RecordingLogger()

    loaded = _load_bmp_efficiency(
        {CFG_BMP_EFFICIENCY: str(efficiency_path)},
        cps=[340],
        pollutants=["TN"],
        logger=logger,
    )

    values = loaded.set_index("pathway")["value"]
    assert values.loc["shallow subsurface"] == pytest.approx(0.0)
    assert values.loc["deep subsurface"] == pytest.approx(0.0)
    assert sum("assuming efficiency=0" in message for message in logger.verbose_messages) == 2


def test_missing_surface_efficiency_rejects_incomplete_cps_pollutant_coverage(tmp_path) -> None:
    efficiency_path = tmp_path / "bmp_efficiency.csv"
    pd.DataFrame(
        [
            {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.25},
            {
                "cps": 340,
                "pollutant": "TP",
                "pathway": "shallow subsurface",
                "value": 0.10,
            },
        ]
    ).to_csv(efficiency_path, index=False)

    with pytest.raises(ValueError) as exc_info:
        _load_bmp_efficiency(
            {CFG_BMP_EFFICIENCY: str(efficiency_path)},
            cps=[340, 590],
            pollutants=["TN", "TP"],
            logger=RecordingLogger(),
        )

    message = str(exc_info.value)
    assert "cps=340, pollutant=TP" in message
    assert "cps=590, pollutant=TN" in message
    assert "cps=590, pollutant=TP" in message


def test_efficiency_sampler_rejects_incomplete_internal_pathway_state() -> None:
    model = SimpleNamespace(
        pollutants=["TN"],
        bmp_efficiency_stats={340: [{"surface": {"value": 0.25}}]},
        _sample_from_stats=lambda stats, kind: float(stats["value"]),
    )

    with pytest.raises(ValueError, match="missing pathways"):
        _sample_efficiency_map(model, 340, 0)
