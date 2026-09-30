import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_config(filename):
    path = ROOT / "config" / filename
    return json.loads(path.read_text(encoding="utf-8"))


def test_settings():
    settings = read_config("settings.json")
    emotions = settings["emotions"]

    assert type(settings["seed"]) is int
    assert type(settings["sample_rate"]) is int
    assert settings["sample_rate"] > 0
    assert len(emotions) == 8
    assert all(isinstance(name, str) and name.strip() for name in emotions)
    assert len(set(emotions)) == 8, "Названия эмоций повторяются"


def test_all_actors_present():
    splits = read_config("splits.json")

    assert set(splits) == {"train", "validation", "test"}

    actors = []
    for group in splits.values():
        assert isinstance(group, list) and group
        assert all(type(actor) is int for actor in group)
        actors.extend(group)

    assert set(actors) == set(range(1, 25)), "Проверь номера дикторов"


def test_no_repeated_actors():
    splits = read_config("splits.json")
    actors = [
        actor
        for group in splits.values()
        for actor in group
    ]

    assert len(actors) == len(set(actors)), (
        "Диктор повторяется внутри выборки или между выборками"
    )