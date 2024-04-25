from typing import Union

import numpy as np

from gymnasium_robotics.envs.robot_env import MujocoPyRobotEnv, MujocoRobotEnv
from gymnasium_robotics.utils import rotations

DEFAULT_CAMERA_CONFIG = {
    # normal
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,

    # "lookat": np.array([1.3, 0.75, 0.55]),
    "lookat": np.array([0.076010, 0.068771, 0.004339]),
}


def zhao_legibility(achieved_goal, goal, distraction_location, t=0, omega=20):
    d_goal = distance(achieved_goal, goal)
    d_dist = distance(achieved_goal, distraction_location)
    return omega * np.exp(-t/30) * np.log(np.abs(d_goal - d_dist) / (d_goal + 1) + 1) * np.sign(d_dist - d_goal)


def relu_at(x, a=0):
    return -1.0 * (x < a)
    # return x * (x < a)


def distance(goal_a, goal_b, view=0, enable_view=False):
    assert goal_a.shape == goal_b.shape
    if enable_view:
        if view.shape == ():
            view = int(view)
            if view == 0:
                goal_a = goal_a[[0, 2]]
                goal_b = goal_b[[0, 2]]
            elif view == 1:
                goal_a = goal_a[[0, 1]]
                goal_b = goal_b[[0, 1]]
            else:
                raise ValueError(
                    'wrong view number, 0 for sideview and 1 for top view')
        else:
            view = view + 1
            view = view.astype(int)
            view = view.reshape(-1, 1)

            rows = np.arange(len(view)).reshape(-1, 1)
            mask = np.ones([len(view), 3])
            mask[rows, view] = False
            mask = mask.astype(bool)

            goal_a = goal_a[mask].reshape(-1, 2)
            goal_b = goal_b[mask].reshape(-1, 2)

    return np.linalg.norm(goal_a - goal_b, axis=-1)


def get_base_xarm6_env(RobotEnvClass: Union[MujocoPyRobotEnv, MujocoRobotEnv]):
    """Factory function that returns a BaseFetchEnv class that inherits
    from MujocoPyRobotEnv or MujocoRobotEnv depending on the mujoco python bindings.
    """

    class BaseXarm6Env(RobotEnvClass):
        """Superclass for all Xarm6 environments."""

        def __init__(
            self,
            gripper_extra_height,
            block_gripper,
            has_object: bool,
            target_in_the_air,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            reward_type,
            distraction,
            viewpoint,
            **kwargs
        ):
            """Initializes a new Fetch environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                gripper_extra_height (float): additional height above the table when positioning the gripper
                block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
                has_object (boolean): whether or not the environment has an object
                target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
                target_offset (float or array with 3 elements): offset of the target
                obj_range (float): range of a uniform distribution for sampling initial object positions
                target_range (float): range of a uniform distribution for sampling a target
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            """

            self.gripper_extra_height = gripper_extra_height
            self.block_gripper = block_gripper
            self.has_object = has_object
            self.target_in_the_air = target_in_the_air
            self.target_offset = target_offset
            self.obj_range = obj_range
            self.target_range = target_range
            self.distance_threshold = distance_threshold
            self.reward_type = reward_type
            self.distraction = distraction
            self.viewpoint = viewpoint

            if 'sample_type' in kwargs.keys():
                sample_type = kwargs.pop('sample_type')
            else:
                sample_type = 'random'

            print(f'sample goal function type: {sample_type}')
            self._sample_goal = self.sample_goal_fn(fn_type=sample_type)

            super().__init__(n_actions=6, **kwargs)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            grip_pos, grip_quat = achieved_goal[...,
                                                :3], achieved_goal[..., 3:7]
            desired_pos, desired_quat, distraction_pos, view = goal[...,
                                                                    :3], goal[..., 3:7], goal[..., 7:10], goal[..., 10]
            attitude_error = rotations.quat_mul(
                rotations.quat_conjugate(grip_quat), desired_quat)
            # attitude_error = rotations.quat_identity()
            rot_error = np.abs(
                2 * np.arccos(np.clip(attitude_error[..., 0], -1., 1.)))
            # import pdb; pdb.set_trace()
            d_goal = distance(grip_pos, desired_pos)
            if self.distraction:
                d_dist = distance(grip_pos, distraction_pos,
                                  view, enable_view=self.viewpoint)
                d_dist = relu_at(d_dist, 0.1)
                reward = -5*d_goal + 3.0 * d_dist - 0.3 * rot_error + \
                    0.3*relu_at(grip_pos[..., 2], 0.0)
            else:
                reward = -d_goal - 1.0*rot_error
            if self.reward_type == "sparse":
                return (d_goal < self.distance_threshold and rot_error < np.pi/9).astype(np.float32)
            if self.reward_type == "dense":
                return reward

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (6,)
            action = action.copy()
            return action

        def _get_obs(self):
            (
                grip_pos,
                grip_quat,
                grip_velp,
                grip_velr,
                robot_qpos,
                robot_qvel,
            ) = self.generate_mujoco_observations()

            achieved_goal = np.concatenate(
                (grip_pos.copy(), grip_quat.copy(), self.goal[7:]))

            obs = np.concatenate(
                [
                    grip_velp,
                    grip_velr,
                    robot_qpos,
                    robot_qvel,
                ]
            )

            return {
                "observation": obs,
                "achieved_goal": achieved_goal,
                "desired_goal": self.goal,
            }

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def sample_goal_fn(self, fn_type='random'):
            if fn_type == 'random':
                def _sample_goal():
                    if self.has_object:
                        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                            -self.target_range, self.target_range, size=3
                        )
                        goal += self.target_offset
                        goal[2] = self.height_offset
                        if self.target_in_the_air and self.np_random.uniform() < 0.5:
                            goal[2] += self.np_random.uniform(0, 0.45)
                    else:
                        # goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                        #     -self.target_range, self.target_range, size=3
                        # )
                        goal = np.array([0.5, 0.0, 0.01]) + np.array([
                            2.5*self.np_random.uniform(-self.target_range,
                                                     self.target_range, size=1)[0],
                            self.np_random.uniform(-self.target_range,
                                                   self.target_range, size=1)[0],
                            0,
                        ])
                        distraction_pos = np.array([0.5, 0.0, 0.01]) + np.array([
                            2.5*self.np_random.uniform(-self.target_range,
                                                     self.target_range, size=1)[0],
                            self.np_random.uniform(-self.target_range,
                                                   self.target_range, size=1)[0],
                            0,
                        ])

                    if self.viewpoint:
                        view = np.random.randint(2, size=1).astype(np.float64)
                    else:
                        view = np.array([0.])
                    return np.concatenate((goal, self.initial_gripper_xquat, distraction_pos, view))

                return _sample_goal

            # side demo
            if fn_type == 'demo1':
                def _sample_goal():
                    goal = np.array([0.6, 0.0, 0.01])
                    distraction_pos = np.array([0.4, 0.0, 0.01])
                    view = np.array([0.])
                    return np.concatenate((goal, self.initial_gripper_xquat, distraction_pos, view))

                return _sample_goal

            # top demo
            if fn_type == 'demo2':
                def _sample_goal():
                    goal = np.array([0.6, 0.0, 0.01])
                    distraction_pos = np.array([0.45, 0.0, 0.01])
                    view = np.array([1.])
                    return np.concatenate((goal, self.initial_gripper_xquat, distraction_pos, view))

                return _sample_goal

            raise ValueError('wrong sample_type')

        def _is_success(self, achieved_goal, goal):
            grip_pos, grip_quat = achieved_goal[...,
                                                :3], achieved_goal[..., 3:7]
            desired_pos, desired_quat, distraction_pos = goal[...,
                                                              :3], goal[..., 3:7], goal[..., 7:10]
            d = distance(grip_pos, desired_pos)
            attitude_error = rotations.quat_mul(
                grip_quat, rotations.quat_conjugate(desired_quat))
            # import pdb; pdb.set_trace()
            rot_error = np.abs(
                2 * np.arccos(np.clip(attitude_error[..., 0], -1., 1.)))
            success = (d < self.distance_threshold).astype(
                np.float32)  # and (rot_error < np.pi / 9).astype(np.float32)
            return success

    return BaseXarm6Env


class MujocoXarm6Env(get_base_xarm6_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):

        if 'sample_type' in kwargs.keys():
            if kwargs['sample_type'] == 'demo1':
                default_camera_config = {
                    "distance": 1.7,
                    "azimuth": 90.0,
                    "elevation": 0.0,

                    "lookat": np.array([0.076010, 0.068771, 0.004339]),
                }
            elif kwargs['sample_type'] == 'demo2':
                default_camera_config = {
                    "distance": 1.7,
                    "azimuth": 180.0,
                    "elevation": -60.0,

                    "lookat": np.array([0.3, 0, 0.004339]),
                }
            else:
                raise ValueError('wrong sample_type')

        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        if self.block_gripper:
            # self._utils.set_joint_qpos(
            #     self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            # )
            # self._utils.set_joint_qpos(
            #     self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            # )
            # self.sim.data.set_joint_qpos('robot0:wrist_roll_joint', 0.)
            # self._mujoco.mj_forward(self.model, self.data)
            pass

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.ctrl_set_action(self.model, self.data, action)
        # self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions
        grip_pos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip")
        grip_mat = self._utils.get_site_xmat(
            self.model, self.data, "robot0:grip")
        # turn mat to quat
        grip_quat = rotations.mat2quat(grip_mat)
        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(
                self.model, self.data, "robot0:grip") * dt
        )
        grip_velr = (
            self._utils.get_site_xvelr(
                self.model, self.data, "robot0:grip") * dt
        )
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        return (
            grip_pos,
            grip_quat,
            grip_velp,
            grip_velr,
            robot_qpos[..., :6],
            robot_qvel[..., :6],
        )

    def _get_gripper_xpos(self):
        body_id = self._model_names.body_name2id["robot0:gripper_link"]
        return self.data.xpos[body_id]

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        self.model.site_pos[site_id] = self.goal[..., :3] - sites_offset[0]
        site_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_SITE, "distraction0"
        )
        self.model.site_pos[site_id] = self.goal[..., 7:10] - sites_offset[0]

        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        # self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        # gripper_target = np.array(
        #     [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        # ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        # gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        # gripper_target = np.array([0., 0., 0. + self.gripper_extra_height]) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        # gripper_target = np.array([0.35, 0.0, 0.05])
        # gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])

        # self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        # self._utils.set_mocap_quat(
        #     self.model, self.data, "robot0:mocap", gripper_rotation
        # )
        # for _ in range(10):
        #     self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()

        initial_gripper_xmat = self._utils.get_site_xmat(
            self.model, self.data, "robot0:grip"
        ).copy()
        # import pdb; pdb.set_trace()
        self.initial_gripper_xquat = rotations.mat2quat(initial_gripper_xmat)
        # self.initial_gripper_xquat = np.array([0, -1., 0, 0])
        if self.has_object:
            self.height_offset = self._utils.get_site_xpos(
                self.model, self.data, "object0")[2]
