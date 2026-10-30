"""CPU execution of the new config class bodies with strict sensor containers.

The constructor fields match the public Isaac Lab v2.2.1 ContactSensorCfg and
SensorBaseCfg declarations. No sensor backend, scene, mesh or policy is created.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_mixed_terrain_config import _column_names, _load_terrain_configs
from test_walking_profiles import _config, _load_profile_module


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"
FEET = ["left_ankle_pitch_link", "right_ankle_pitch_link"]
JOINTS = [f"joint_{index}" for index in range(29)]


class _ContactSensorCfg:
    """Strict keyword seam for official v2.2.1 sensor configuration fields."""

    def __init__(self, *, prim_path, filter_prim_paths_expr, history_length,
                 update_period, max_contact_data_count_per_prim, debug_vis=False):
        self.prim_path = prim_path
        self.filter_prim_paths_expr = list(filter_prim_paths_expr)
        self.history_length = history_length
        self.update_period = update_period
        self.max_contact_data_count_per_prim = max_contact_data_count_per_prim
        self.debug_vis = debug_vis


class _RewardTermCfg:
    def __init__(self, *, func, weight, params=None):
        self.func, self.weight = func, weight
        self.params = {} if params is None else dict(params)


class _RobotSceneCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(deepcopy(_config().scene.__dict__))
        self.__dict__.update(kwargs)
        self.terrain.prim_path = "/World/ground"
        self.contact_forces = SimpleNamespace(identity="original_all_body_sensor")


class _RobotStairsEnvCfg:
    scene = _RobotSceneCfg(num_envs=2048, env_spacing=3.5)
    feet_body_names = FEET

    def __init__(self):
        scene = deepcopy(type(self).scene)
        self.__dict__.update(deepcopy(_config().__dict__))
        self.scene = scene
        self.sim = SimpleNamespace(dt=.005)
        self.parent_post_init_calls = 0

    def __post_init__(self):
        self.parent_post_init_calls += 1


def _load_configs():
    source = TASK / "walking_env_cfg.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    terrain = _load_terrain_configs()
    namespace = {
        "RobotSceneCfg": _RobotSceneCfg, "RobotStairsEnvCfg": _RobotStairsEnvCfg,
        "ContactSensorCfg": _ContactSensorCfg, "configclass": lambda cls: cls,
        "H2_JOINT_NAMES": JOINTS, "H2_FOOT_BODY_NAMES": FEET,
        "MIXED_WALK_TERRAINS_CFG": terrain["MIXED_WALK_TERRAINS_CFG"],
        "STAIR_ROUGH_TERRAINS_CFG": terrain["STAIR_ROUGH_TERRAINS_CFG"],
        "apply_walking_profile": _load_profile_module().apply_walking_profile,
        "RewTerm": _RewardTermCfg,
        "h2_mdp": SimpleNamespace(walk_signed_progress=abs, walk_alternating_steps=sum, walk_support_quality=min),
        "WALK_REWARD_PARAMETERS": _load_profile_module().WALK_REWARD_PARAMETERS,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), namespace)
    return namespace


def _new_cfg(namespace, name):
    cfg = namespace[name]()
    cfg.__post_init__()
    return cfg


def test_scene_declares_opt_in_sensor_fields_without_changing_base_scene():
    namespace = _load_configs()
    scene_class = namespace["WalkingSceneCfg"]
    for name in ("pelvis_ground", "pelvis_robot", "left_foot_ground", "right_foot_ground"):
        assert name in scene_class.__annotations__
        assert getattr(scene_class, name) is None
        assert not hasattr(_RobotSceneCfg, name)


@pytest.mark.parametrize("name", ["RobotContactDiagnosticEnvCfg", "RobotContactDiagnosticPlayEnvCfg"])
def test_contact_diagnostics_has_four_single_body_filters_at_physics_rate(name):
    namespace = _load_configs()
    cfg = _new_cfg(namespace, name)
    assert cfg.contact_source_diagnostics is True
    assert cfg.staged_evidence is False
    assert cfg.walking_diagnostic_joint_names == tuple(JOINTS)
    assert cfg.contact_diagnostic_threshold == 1.
    assert cfg.contact_diagnostic_max_samples == 8
    assert cfg.stage_support_height_tolerance == .04
    expected = {
        "pelvis_ground": ("pelvis", ["/World/ground/terrain/mesh"]),
        "pelvis_robot": ("pelvis", ["{ENV_REGEX_NS}/Robot/" + body
                                      for body in namespace["H2_DIAGNOSTIC_COLLISION_BODY_NAMES"] if body != "pelvis"]),
        "left_foot_ground": (FEET[0], ["/World/ground/terrain/mesh"]),
        "right_foot_ground": (FEET[1], ["/World/ground/terrain/mesh"]),
    }
    for key, (body, filters) in expected.items():
        sensor = getattr(cfg.scene, key)
        assert sensor.prim_path == "{ENV_REGEX_NS}/Robot/" + body
        assert sensor.filter_prim_paths_expr == filters
        assert sensor.history_length == 3
        assert sensor.update_period == cfg.sim.dt
        assert sensor.max_contact_data_count_per_prim == 64
    assert cfg.scene.contact_forces.identity == "original_all_body_sensor"


@pytest.mark.parametrize("diagnostics,staged", [(False, False), (True, False), (False, True), (True, True)])
def test_filtered_sensors_are_enabled_only_when_a_consumer_requires_them(diagnostics, staged):
    namespace = _load_configs()
    cfg = namespace["RobotContactDiagnosticEnvCfg"]()
    cfg.contact_source_diagnostics = diagnostics
    cfg.staged_evidence = staged
    cfg.__post_init__()
    for name in ("pelvis_ground", "pelvis_robot", "left_foot_ground", "right_foot_ground"):
        assert (getattr(cfg.scene, name) is not None) is (diagnostics or staged)


def test_diagnostic_only_configuration_preserves_original_mdp_components():
    namespace = _load_configs()
    cfg = namespace["RobotContactDiagnosticEnvCfg"]()
    original = {name: getattr(cfg, name) for name in ("actions", "rewards", "terminations", "commands", "observations", "curriculum", "events")}
    cfg.__post_init__()
    assert cfg.parent_post_init_calls == 1
    assert all(getattr(cfg, name) is value for name, value in original.items())
    assert cfg.events.push_robot is not None
    assert cfg.observations.policy.enable_corruption is True
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-.1, .3)


@pytest.mark.parametrize("stem,stage", [
    ("RobotWalkBootstrap", "bootstrap"), ("RobotWalkShallow", "shallow"),
    ("RobotWalkStairs08", "stairs_08"), ("RobotWalkStairs12", "stairs_12"),
    ("RobotWalkStairs16", "stairs_16"), ("RobotWalkStairs20", "stairs_20"),
])
def test_explicit_stage_train_play_share_physical_command_and_scan_semantics(stem, stage):
    namespace = _load_configs()
    train = _new_cfg(namespace, stem + "EnvCfg")
    play = _new_cfg(namespace, stem + "PlayEnvCfg")
    assert train.parent_post_init_calls == play.parent_post_init_calls == 1
    assert train.walking_stage == play.walking_stage == stage
    assert train.contact_source_diagnostics is play.contact_source_diagnostics is True
    assert train.staged_evidence is play.staged_evidence is True
    assert play.scene.num_envs == 64
    for cfg in (train, play):
        assert cfg.scene.height_scanner is not None
        assert cfg.stage_support_height_tolerance < .02
        assert cfg.observations.policy.height_scanner is not None
        assert cfg.observations.critic.height_scanner is not None
        assert cfg.use_lidar_reconstruction is False
        assert cfg.commands.base_velocity.ranges.lin_vel_x == (.2, .2)
        assert cfg.commands.base_velocity.ranges.lin_vel_y == (0., 0.)
        assert cfg.commands.base_velocity.ranges.ang_vel_z == (0., 0.)
        assert cfg.commands.base_velocity.heading_command is False
        assert cfg.events.reset_base.params["pose_range"]["yaw"] == (0., 0.)
        assert cfg.events.push_robot is None
        assert cfg.events.physics_material is None
        assert cfg.events.add_motor_pd_and_motor_strength is None
        assert cfg.observations.policy.enable_corruption is False
        assert cfg.scene.robot.actuators["leg"].max_delay == 0
    assert train.scene.terrain.terrain_generator == play.scene.terrain.terrain_generator
    families = ["flat"] if stage == "bootstrap" else ["flat"] * 6 + ["up"] * 2 + ["down"] * 2
    assert _column_names(train.scene.terrain.terrain_generator) == families


def test_uphill_uses_inverted_pyramid_and_downhill_regular_pyramid():
    namespace = _load_configs()
    # Factories in the terrain test use identical containers; attach a distinct
    # sentinel to the real source template data to check actual wiring.
    namespace["STAIR_ROUGH_TERRAINS_CFG"].sub_terrains["stairs_down"].physical_type = "inverted"
    namespace["STAIR_ROUGH_TERRAINS_CFG"].sub_terrains["stairs_low"].physical_type = "regular"
    cfg = _new_cfg(namespace, "RobotWalkShallowEnvCfg")
    generated = cfg.scene.terrain.terrain_generator
    assert generated.sub_terrains["up"].physical_type == "inverted"
    assert generated.sub_terrains["down"].physical_type == "regular"


def test_profile_ablation_switches_reach_helper_and_train_play_agree():
    namespace = _load_configs()
    configs = []
    for name in ("RobotWalkBootstrapEnvCfg", "RobotWalkBootstrapPlayEnvCfg"):
        cfg = namespace[name]()
        cfg.walking_enable_push = True
        cfg.walking_enable_domain_randomization = True
        cfg.walking_enable_observation_noise = True
        cfg.walking_enable_actuator_delay = True
        cfg.walking_forward_speed = .15
        cfg.__post_init__()
        configs.append(cfg)
        assert cfg.events.push_robot is not None
        assert cfg.events.physics_material is not None
        assert cfg.events.add_motor_pd_and_motor_strength is not None
        assert cfg.observations.policy.enable_corruption is True
        assert cfg.scene.robot.actuators["leg"].max_delay == 6
        assert cfg.commands.base_velocity.ranges.lin_vel_x == (.15, .15)
    assert configs[0].scene.terrain.terrain_generator == configs[1].scene.terrain.terrain_generator


def test_config_instances_do_not_share_sensor_objects():
    namespace = _load_configs()
    first = _new_cfg(namespace, "RobotContactDiagnosticEnvCfg")
    second = _new_cfg(namespace, "RobotContactDiagnosticEnvCfg")
    assert first.scene is not second.scene
    assert first.scene.pelvis_ground is not second.scene.pelvis_ground
    first.scene.pelvis_ground.filter_prim_paths_expr.append("unrelated")
    assert second.scene.pelvis_ground.filter_prim_paths_expr == ["/World/ground/terrain/mesh"]


@pytest.mark.parametrize("flag", ["contact_source_diagnostics", "staged_evidence"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_diagnostic_consumer_flags_require_real_booleans(flag, value):
    namespace = _load_configs()
    cfg = namespace["RobotContactDiagnosticEnvCfg"]()
    setattr(cfg, flag, value)
    with pytest.raises(ValueError, match="booleans"):
        cfg.__post_init__()


def _matches_sensor_glob(expression, prim_paths):
    # The official ContactSensor converts regex-style .* into PhysX globs.
    glob = expression.replace("{ENV_REGEX_NS}", "/World/envs/env_.*").replace(".*", "*")
    return [path for path in prim_paths if fnmatchcase(path, glob)]


def test_self_contact_filters_match_one_body_per_environment_instead_of_all_links():
    namespace = _load_configs()
    cfg = _new_cfg(namespace, "RobotContactDiagnosticEnvCfg")
    bodies = namespace["H2_DIAGNOSTIC_COLLISION_BODY_NAMES"]
    prims = [f"/World/envs/env_{env}/Robot/{body}" for env in range(2) for body in bodies]
    primary_count = len(_matches_sensor_glob(cfg.scene.pelvis_robot.prim_path, prims))
    assert primary_count == 2
    filters = cfg.scene.pelvis_robot.filter_prim_paths_expr
    assert len(filters) == 22
    matches = [_matches_sensor_glob(expression, prims) for expression in filters]
    assert all(len(paths) == primary_count for paths in matches)
    assert len({path for paths in matches for path in paths}) == 2 * 22
    assert all(not path.endswith("/pelvis") for paths in matches for path in paths)
    # Regression: a single broad expression produces N*B matches, violating
    # the per-filter N-clone cardinality used by batched filtered contacts.
    old_matches = _matches_sensor_glob("{ENV_REGEX_NS}/Robot/.*", prims)
    assert len(old_matches) == 2 * 23
    assert len(old_matches) != primary_count


def test_ground_filter_resolves_the_colliding_child_created_by_official_mesh_seam():
    namespace = _load_configs()
    cfg = _new_cfg(namespace, "RobotContactDiagnosticEnvCfg")
    created, collisions = {}, []

    def create_prim(path, prim_type, **kwargs):
        created[path] = prim_type
        return SimpleNamespace(GetPrimPath=lambda: path)

    # Frozen executable statements from Isaac Lab v2.2.1 terrains/utils.py
    # create_prim_from_mesh, lines 79–95. Imports/material/color handling are
    # intentionally outside this collision-path seam and are not executed.
    source = '''
def create_prim_from_mesh(prim_path, mesh, **kwargs):
    prim_utils.create_prim(prim_path, "Xform")
    prim = prim_utils.create_prim(
        f"{prim_path}/mesh", "Mesh", translation=kwargs.get("translation"),
        orientation=kwargs.get("orientation"), attributes={
            "points": mesh.vertices, "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)), "subdivisionScheme": "bilinear"})
    collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)
'''
    scope = {
        "prim_utils": SimpleNamespace(create_prim=create_prim), "np": np,
        "sim_utils": SimpleNamespace(CollisionPropertiesCfg=SimpleNamespace,
                                      define_collision_properties=lambda path, collider: collisions.append(path)
                                      if collider.collision_enabled else None),
    }
    exec(compile(ast.parse(source), "IsaacLab-v2.2.1-create_prim_from_mesh-collision-seam", "exec"), scope)
    mesh = SimpleNamespace(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    # TerrainImporter imports generator output under name="terrain".
    scope["create_prim_from_mesh"](cfg.scene.terrain.prim_path + "/terrain", mesh)
    ground_filter = cfg.scene.pelvis_ground.filter_prim_paths_expr[0]
    assert _matches_sensor_glob(ground_filter, collisions) == collisions
    assert created[ground_filter] == "Mesh"
    assert _matches_sensor_glob(cfg.scene.terrain.prim_path, collisions) == []


@pytest.mark.parametrize("terrain_type", ["plane", "usd", None])
def test_active_filters_reject_unsupported_terrain_import_protocol(terrain_type):
    namespace = _load_configs()
    cfg = namespace["RobotContactDiagnosticEnvCfg"]()
    cfg.scene.terrain.terrain_type = terrain_type
    with pytest.raises(ValueError, match="generator"):
        cfg.__post_init__()


def test_ground_collision_filter_follows_a_valid_custom_terrain_prim_path():
    namespace = _load_configs()
    cfg = namespace["RobotContactDiagnosticEnvCfg"]()
    cfg.scene.terrain.prim_path = "/World/custom_ground"
    cfg.__post_init__()
    assert cfg.scene.pelvis_ground.filter_prim_paths_expr == ["/World/custom_ground/terrain/mesh"]


@pytest.mark.parametrize("path", ["World/ground", "/World/ground/", "/World/.*", "/World//ground", None])
def test_active_filters_reject_ambiguous_terrain_prim_paths(path):
    namespace = _load_configs()
    cfg = namespace["RobotContactDiagnosticEnvCfg"]()
    cfg.scene.terrain.prim_path = path
    with pytest.raises(ValueError, match="prim_path"):
        cfg.__post_init__()


def test_disabled_filtered_diagnostics_does_not_impose_a_terrain_protocol():
    namespace = _load_configs()
    cfg = namespace["RobotContactDiagnosticEnvCfg"]()
    cfg.contact_source_diagnostics = False
    cfg.staged_evidence = False
    cfg.scene.terrain.terrain_type = "usd"
    cfg.scene.terrain.prim_path = None
    cfg.__post_init__()
    assert cfg.scene.pelvis_ground is cfg.scene.pelvis_robot is None
    assert cfg.scene.left_foot_ground is cfg.scene.right_foot_ground is None


@pytest.mark.parametrize("stem", [
    "RobotWalkBootstrap", "RobotWalkShallow", "RobotWalkStairs08",
    "RobotWalkStairs12", "RobotWalkStairs16", "RobotWalkStairs20",
])
@pytest.mark.parametrize("play", [False, True])
def test_new_walk_configs_select_bounded_timing_and_preserve_legacy_diagnostic(stem, play):
    namespace = _load_configs()
    cfg = _new_cfg(namespace, stem + ("PlayEnvCfg" if play else "EnvCfg"))
    assert cfg.walking_cross_time_mode == "bounded"
    assert cfg.walking_max_time_difference == .5
    assert cfg.rewards.feet_air_time_variance.params["cross_time_mode"] == "bounded"
    assert cfg.rewards.feet_air_time_variance.params["max_time_difference"] == .5
    legacy = _new_cfg(namespace, "RobotContactDiagnosticEnvCfg")
    assert "cross_time_mode" not in legacy.rewards.feet_air_time_variance.params
    assert "max_time_difference" not in legacy.rewards.feet_air_time_variance.params


@pytest.mark.parametrize("play", [False, True])
def test_legacy_cross_time_ablation_is_an_explicit_fresh_config_choice(play):
    namespace = _load_configs()
    cfg = namespace["RobotWalkBootstrap" + ("PlayEnvCfg" if play else "EnvCfg")]()
    cfg.walking_cross_time_mode = "legacy"
    cfg.walking_max_time_difference = .3
    cfg.__post_init__()
    assert cfg.rewards.feet_air_time_variance.params["cross_time_mode"] == "legacy"
    assert cfg.rewards.feet_air_time_variance.params["max_time_difference"] == .3

