'''
#CHER part
python train.py --env_name 'FetchReach-v1' --logdir 'fetchreachv1/cpu1ep50/alg=DDPG+CHER=/r0'
python train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50/alg=DDPG+CHER=/r0'
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50/alg=DDPG+CHER=/r0'
python train.py --env_name 'HandManipulateBlockRotateXYZ-v0' --logdir 'HandManipulateBlockRotateXYZ-v0/cpu1ep50/alg=DDPG+CHER=/r0'
python train.py --env_name 'HandManipulatePenRotate-v0' --logdir 'HandManipulatePenRotate-v0/cpu1ep50/alg=DDPG+CHER=/r0'
'''

'''
#HER part
python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'FetchReach-v1' --logdir 'fetchreachv1/cpu1ep50/alg=DDPG+HER=/r1'
python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50/alg=DDPG+HER=/r1'
python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50/alg=DDPG+HER=/r1'
python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandManipulateBlockRotateXYZ-v0' --logdir 'HandManipulateBlockRotateXYZ-v0/cpu1ep50/alg=DDPG+HER=/r1'
python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandManipulatePenRotate-v0' --logdir 'HandManipulatePenRotate-v0/cpu1ep50/alg=DDPG+HER=/r1'
'''

'''
#lib buffer
#python train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50buffer1E7/alg=DDPG+CHER=/r0' --lib_buffer 10000000
#python train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50buffer1E5/alg=DDPG+CHER=/r0' --lib_buffer 100000
#python train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50buffer1E4/alg=DDPG+CHER=/r0' --lib_buffer 10000
#python train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50buffer1E3/alg=DDPG+CHER=/r0' --lib_buffer 1000
'''


'''
#more epoch
python train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50epochs1000/alg=DDPG+CHER=/r0' --n_epochs 1000
'''

# ===========LAB 2============================================================================
# more cpu parallel
'''
#python train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50/alg=DDPG+CHER=/r2' --num_cpu 10
#python train.py --env_name 'FetchReach-v1' --logdir 'fetchreachv1/cpu1ep50/alg=DDPG+CHER=/r2' --num_cpu 10
#python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50/alg=DDPG+CHER=/r2' --num_cpu 10
#python train.py --env_name 'HandManipulateBlockRotateXYZ-v0' --logdir 'HandManipulateBlockRotateXYZ-v0/cpu1ep50/alg=DDPG+CHER=/r2' --num_cpu 10
#python train.py --env_name 'HandManipulatePenRotate-v0' --logdir 'HandManipulatePenRotate-v0/cpu1ep50/alg=DDPG+CHER=/r2' --num_cpu 10

#python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'FetchReach-v1' --logdir 'fetchreachv1/cpu1ep50/alg=DDPG+HER=/r2' --num_cpu 10
#python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandReach-v0' --logdir 'HandReach-v0/cpu1ep50/alg=DDPG+HER=/r2' --num_cpu 10
#python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50/alg=DDPG+HER=/r2' --num_cpu 10
#python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandManipulateBlockRotateXYZ-v0' --logdir 'HandManipulateBlockRotateXYZ-v0/cpu1ep50/alg=DDPG+HER=/r2' --num_cpu 10
#python /home/jeff/CHER/baselines/her/experiment/train.py --env_name 'HandManipulatePenRotate-v0' --logdir 'HandManipulatePenRotate-v0/cpu1ep50/alg=DDPG+HER=/r2' --num_cpu 10
'''

'''
#test num_cpu effect
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50/alg=DDPG+CHER=/r3' --num_cpu 12

#batch size experiment

#buffer size experiment
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50buffer1E7/alg=DDPG+CHER=/r2' --lib_buffer 10000000 --num_cpu 10
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50buffer1E5/alg=DDPG+CHER=/r2' --lib_buffer 100000 --num_cpu 10
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50buffer1E4/alg=DDPG+CHER=/r2' --lib_buffer 10000 --num_cpu 10
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50buffer1E3/alg=DDPG+CHER=/r2' --lib_buffer 1000 --num_cpu 10
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50buffer1E2/alg=DDPG+CHER=/r2' --lib_buffer 100 --num_cpu 10

#epochs experiment
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50epochs1000/alg=DDPG+CHER=/r2' --n_epochs 1000  --num_cpu 10

'''

#replay k experiment
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk1/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 1
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk2/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 2
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk3/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 3
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk5/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 5
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk6/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 6
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk7/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 7
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk8/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 8
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk9/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 9
python train.py --env_name 'HandManipulateEggFull-v0' --logdir 'HandManipulateEggFull-v0/cpu1ep50replayk10/alg=DDPG+CHER=/r2' --num_cpu 12 --replay_k 10

