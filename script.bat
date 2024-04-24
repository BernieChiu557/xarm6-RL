@REM training
@REM Fetch robot
python -m rl_zoo3.train --algo sac --env FetchReach-v2 --eval-freq 2000 --seed 42 --conf-file her.yaml --wandb-project-name FetcReach --track  
@REM xArm
@REM [Sparse, Dense], [Dist, NoDist] 
python -m rl_zoo3.train --algo sac --gym-packages gymnasium_xarm6 --env xArm6ReachSparseDist-v1 --eval-freq 2000 --seed 42 --conf-file config.yaml --wandb-project-name XArmReach --track  
python -m rl_zoo3.train --algo sac --gym-packages gymnasium_xarm6 --env xArm6ReachSparseNoDist-v1 --eval-freq 2000 --seed 42 --conf-file config.yaml --wandb-project-name XArmReach --track  
python -m rl_zoo3.train --algo sac --gym-packages gymnasium_xarm6 --env xArm6ReachDenseDist-v1 --eval-freq 2000 --seed 42 --conf-file config.yaml --wandb-project-name XArmReach --track  
python -m rl_zoo3.train --algo sac --gym-packages gymnasium_xarm6 --env xArm6ReachDenseNoDist-v1 --eval-freq 2000 --seed 42 --conf-file config.yaml --wandb-project-name XArmReach --track  
python -m rl_zoo3.train --algo sac --gym-packages gymnasium_xarm6 --env xArm6ReachDenseDistView-v1 --eval-freq 2000 --seed 1 --conf-file config.yaml --wandb-project-name XArmReach --track

@REM make video
python record_video.py -f logs --env FetchReach-v2 --algo sac --exp-id 3
python record_video.py -f logs --env xArm6ReachDenseDist-v1 --gym-packages gymnasium_xarm6 --algo sac --exp-id 4
python record_video.py -f logs --env xArm6ReachDenseDistView-v1 --gym-packages gymnasium_xarm6 --algo sac --exp-id 10 --env-kwargs sample_type:'demo1' -n 100