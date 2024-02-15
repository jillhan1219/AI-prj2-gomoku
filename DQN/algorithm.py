from DQN import DQN
import numpy as np
import os
import math
import copy

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

CHECKPOINT = 40000  # 使用训练了CHECKPOINT轮的网络存档

def init_robot():
    # initialize DQN
    agent = DQN()
    agent.copy()

    # load checkpoints
    checkpoint_dir = './checkpoints'
    checkpoint_step = 'model.ckpt-' + str(CHECKPOINT)

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_step)
    agent.load_checkpoint(checkpoint_path)
    
    return agent

def robot(chessboard, robot_color, last_drop, agent):
    # chessboard    a 15*15 ndarray, 0 is empty, -1 is black, 1 is white
    # robot_color   a interger, -1 is black, 1 is white
    # last_drop     a tuple (r,c), r is row of chessboard, c is column of chessboard

    # RETURN:       a tuple (r,c), which is location of robot to drop piece in this turn

    SIZE=15
    camp = np.zeros([1])
    camp[0] = robot_color

    # state = chessboard
    state = copy.deepcopy(chessboard)

    # 将state中的0、-1和1转化为None、0和1
    for i in range(SIZE):
        for j in range(SIZE):
            if state[i][j] == 0:
                state[i][j] = 2
            elif state[i][j] == -1:
                state[i][j] = 0
            else:
                state[i][j] = 1

    # print("state: ", state)

    state = np.reshape(state, [-1])
    state = [state, camp]

    action = agent.action(state)  # 按照Q网络走一步
    action = [math.floor(action/SIZE), action%SIZE, camp]  # 转化为二维棋盘坐标

    r, c = action[0], action[1]

    # print("robot drop: ", (r, c))
    return (r, c)