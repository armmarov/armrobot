import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

PM01_URDF_PATH = os.path.join(os.path.dirname(__file__), "pm01_assets", "urdf", "pm01_only_legs_simple_collision.urdf")

PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=PM01_URDF_PATH,
        fix_base=False,
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "j00_hip_pitch_l": -0.24,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.48,
            "j04_ankle_pitch_l": -0.24,
            "j05_ankle_roll_l": 0.0,
            "j06_hip_pitch_r": -0.24,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.48,
            "j10_ankle_pitch_r": -0.24,
            "j11_ankle_roll_r": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["j00_hip_pitch_l", "j06_hip_pitch_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=70.0,
            damping=7.0,
            armature=0.0453,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["j01_hip_roll_l", "j07_hip_roll_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=50.0,
            damping=5.0,
            armature=0.0453,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["j02_hip_yaw_l", "j08_hip_yaw_r"],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=50.0,
            damping=5.0,
            armature=0.0067,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["j03_knee_pitch_l", "j09_knee_pitch_r"],
            effort_limit_sim=164.0,
            velocity_limit_sim=26.3,
            stiffness=70.0,
            damping=7.0,
            armature=0.0453,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[
                "j04_ankle_pitch_l",
                "j05_ankle_roll_l",
                "j10_ankle_pitch_r",
                "j11_ankle_roll_r",
            ],
            effort_limit_sim=52.0,
            velocity_limit_sim=35.2,
            stiffness=20.0,
            damping=0.2,
            armature=0.0067,
        ),
        # Upper body joints (j12-j23) are fixed in the legs-only URDF — no actuators needed
    },
)
