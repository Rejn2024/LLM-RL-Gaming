from war_game.engine import GameEngine
from war_game.map import HexMap
from war_game.model import Action, Faction, Hex, Terrain, Unit


def test_hex_distance_and_path_around_water():
    game_map = HexMap(4, 4, {Hex(1, 0): Terrain.WATER})
    assert Hex(0, 0).distance(Hex(2, 1)) == 3
    path = game_map.path(Hex(0, 0), Hex(2, 0))
    assert path[-1] == Hex(2, 0)
    assert Hex(1, 0) not in path


def test_combat_ew_and_fog_of_war():
    blue = Unit("blue", Faction.BLUE, Hex(0, 0), attack_range=2, sensor_range=3, ew_power=1.0)
    red = Unit("red", Faction.RED, Hex(1, 0))
    hidden = Unit("hidden", Faction.RED, Hex(5, 5))
    engine = GameEngine(HexMap(6, 6), [blue, red, hidden], seed=1)
    assert {u.id for u in engine.visible_units(Faction.BLUE)} == {"blue", "red"}
    engine.submit(Action("blue", "ew", target_id="red")); engine.step(2)
    assert red.jammed_for > 0
    blue.cooldown = 0
    engine.submit(Action("blue", "attack", target_id="red")); engine.step()
    assert red.hp < 100


def test_white_force_does_not_prevent_victory():
    engine = GameEngine(HexMap(2, 2), [Unit("blue", Faction.BLUE, Hex(0, 0)),
                                      Unit("white", Faction.WHITE, Hex(1, 1), attack=0)])
    assert engine.winner() == Faction.BLUE


def test_step_returns_new_events_and_movement_stops_before_occupied_cell():
    blue = Unit("blue", Faction.BLUE, Hex(0, 0), movement=3)
    red = Unit("red", Faction.RED, Hex(2, 0))
    engine = GameEngine(HexMap(4, 2), [blue, red])
    engine.submit(Action("blue", "move", target_hex=Hex(1, 0)))
    events = engine.step()
    assert [event.kind for event in events] == ["move"]
    assert engine.step() == []
    engine.submit(Action("blue", "move", target_hex=red.position))
    engine.step(2)
    assert blue.position != red.position
