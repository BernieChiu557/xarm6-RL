import os
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_xarm6.envs.xarm6_env import MujocoXarm6Env


class xArm6ReachEnv(MujocoXarm6Env, EzPickle):
    def __init__(
            self, 
            reward_type: str = "sparse", 
            distraction: bool = True, 
            viewpoint: bool = True, 
            **kwargs):

        # Ensure we get the path separator correct on windows
        if distraction:
            MODEL_XML_PATH = os.path.join('assets', 'reach_with_distraction.xml')
        else:
            MODEL_XML_PATH = os.path.join('assets', 'reach.xml')
        fullpath = os.path.join(os.path.dirname(__file__), MODEL_XML_PATH)
        # print(reward_type)
        # print(distraction)
        # print(fullpath)

        initial_qpos = {
            'robot0:slide0': 0.,
            'robot0:slide1': 0.,
            'robot0:slide2': 0.,
            "robot0:shoulder_pan_joint": 0.0,
            "robot0:shoulder_lift_joint": -1.0,
            "robot0:elbow_flex_joint": 0.0,
            "robot0:forearm_roll_joint": 0.0,
            "robot0:wrist_flex_joint": 1.0,
            "robot0:wrist_roll_joint": 0.0,
        }

        print(f'enable viewpoint: {viewpoint}')
        MujocoXarm6Env.__init__(
            self, 
            model_path=fullpath, 
            has_object=False, 
            block_gripper=True, 
            # n_substeps=30,
            n_substeps=20,
            gripper_extra_height=0.0, 
            target_in_the_air=True, 
            target_offset=0.0,
            obj_range=0.25, 
            target_range=0.1, 
            distance_threshold=0.01,
            initial_qpos=initial_qpos, 
            reward_type=reward_type,
            distraction=distraction,
            viewpoint=viewpoint,
            **kwargs)
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)
