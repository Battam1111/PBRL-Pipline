import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)

from scene_point_cloud_processor import ScenePointCloudExtractor, PointCloudSaver

class SawyerNutDisassembleEnvV2(SawyerXYZEnv):
    WRENCH_HANDLE_LENGTH = 0.02

    def __init__(self, tasks=None, render_mode=None):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.6, 0.025)
        obj_high = (0.1, 0.75, 0.02501)
        goal_low = (-0.1, 0.6, 0.1699)
        goal_high = (0.1, 0.75, 0.1701)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.visualizer = PointCloudSaver()  # 初始化保存器

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.7, 0.025]),
            "hand_init_pos": np.array((0, 0.4, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.8, 0.17])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0.0, 0.0, 0.005]),
            np.array(goal_high) + np.array([0.0, 0.0, 0.005]),
        )

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_assembly_peg.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(success),
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [("pegTop", self._target_pos)]

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("WrenchHandle")

    def _get_pos_objects(self):
        return self._get_site_pos("RoundNut-8")

    def _get_quat_objects(self):
        return self.data.body("RoundNut").xquat

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self.get_body_com("RoundNut")
        return obs_dict

    # 标记：pointcloud
    def render(self, mode=''):
        """
        渲染环境。

        参数：
            mode (str): 渲染模式，可以是 'human'、'rgb_array'、'depth_array' 或 'pointcloud'。
            width (int, 可选): 渲染宽度。如果为 None，则使用环境的默认宽度。
            height (int, 可选): 渲染高度。如果为 None，则使用环境的默认高度。
            camera_name (str): 摄像机名称。

        返回：
            如果 mode 为 'pointcloud'，返回完整的场景点云数据；否则，返回渲染结果。
        """
        if mode == 'pointcloud':
            extractor = ScenePointCloudExtractor(
                self.model, 
                self.data, 
                task_related_body_names=["RoundNut", "peg", "hand"]
                )
            point_cloud = extractor.extract_point_cloud()

            # self.visualizer.save_point_cloud(point_cloud)  # 保存点云文件
            return point_cloud
        else:
            return super().render()

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]
        self._target_pos = goal_pos[:3] + np.array([0, 0, 0.15])

        peg_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.03])
        peg_top_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.08])
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        ] = peg_pos
        self.model.site_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pegTop")
        ] = peg_top_pos
        mujoco.mj_forward(self.model, self.data)
        self._set_obj_xyz(self.obj_init_pos)
        return self._get_obs()

    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos + np.array([0.0, 0.0, 0.1]) - wrench_center

        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid="long_tail",
        )

        return in_place

    def compute_reward(self, actions, obs):
        hand = obs[:3]
        wrench = obs[4:7]
        wrench_center = self._get_site_pos("RoundNut")
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the wrench handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified wrench position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        wrench_threshed = wrench.copy()
        threshold = SawyerNutDisassembleEnvV2.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        reward_quat = SawyerNutDisassembleEnvV2._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions,
            wrench_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            high_density=True,
        )
        reward_in_place = SawyerNutDisassembleEnvV2._reward_pos(
            wrench_center, self._target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        # Override reward on success
        success = obs[6] > self._target_pos[2]
        if success:
            reward = 10.0

        return (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            success,
        )


class TrainDisassemblev2(SawyerNutDisassembleEnvV2):
    tasks = None

    def __init__(self):
        SawyerNutDisassembleEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestDisassemblev2(SawyerNutDisassembleEnvV2):
    tasks = None

    def __init__(self):
        SawyerNutDisassembleEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
