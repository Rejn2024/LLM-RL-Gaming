from war_game.engine import GameEngine
from war_game.llm import actions_from_plan, observation
from war_game.model import Faction, Hex
from war_game.scenario import demo_scenario


def test_observation_and_plan_validation():
    game_map, units = demo_scenario()
    engine = GameEngine(game_map, units)
    state = observation(engine, Faction.BLUE)
    assert all(u["id"] != "red-1" for u in state["units"])
    plan = {"tasks": [
        {"actor_id": "blue-1", "action": "move", "target_hex": {"q": 2, "r": 2}},
        {"actor_id": "red-1", "action": "attack", "target_id": "blue-1"},
        {"actor_id": "blue-1", "action": "move", "target_hex": {"q": 999, "r": 2}},
    ]}
    actions = actions_from_plan(plan, engine, Faction.BLUE)
    assert len(actions) == 1 and actions[0].target_hex == Hex(2, 2)
