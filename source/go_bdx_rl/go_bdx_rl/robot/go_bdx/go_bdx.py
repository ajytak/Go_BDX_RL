import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from whole_body_tracking.assets import ASSET_DIR

##
# Configuration
##

GO_BDX = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/workspace/isaac_projects/go_bdx_rl/source/go_bdx_rl/go_bdx_rl/robot/go_bdx/go_bdx.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        fix_base = False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.00),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "bdx": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*_hip.*": 50.0,
                ".*_knee": 50.0,
                ".*_ankle": 50.0,
                # "neck_pitch": 100.0,
                # "head.*": 100.0,
                
            },
            damping={
                ".*_hip.*": 1.0,
                ".*_knee": 1.0,
                ".*_ankle": 1.0,
                # "neck_pitch": 1.0,
                # "head.*": 1.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)