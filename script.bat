@REM training
@REM python -m rl_zoo3.train --algo sac --env FetchReach-v2 --eval-freq 2000 --seed 42 --conf-file her.yaml --wandb-project-name FetcReach --track  
@REM python -m rl_zoo3.train --algo sac --gym-packages gymnasium_xarm6 --env xArm6Reach-v1 --eval-freq 2000 --seed 42 --conf-file her.yaml --wandb-project-name XArmReach --track  


@REM make video
@REM python record_video.py -f logs --env FetchReach-v2 --algo sac --exp-id 3
@REM python record_video.py -f logs --env xArm6Reach-v1 --gym-packages gymnasium_xarm6 --algo sac --exp-id 4

stable baselline
torch
RL-zoo
gymnasium
gymnasium-robotics
mujoco