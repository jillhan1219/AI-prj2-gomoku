EMPTY = 2
GAMMA = 0.9        
INITIAL_E = 0.1   
FINAL_E = 0.001		
REPLAY_NUM = 15000  
BATCH_NUM = 200  	
Q_STEP = 100  

IS_TRAIN = False   # 训练 or 测试
CHECKPOINT = 1000  # 使用训练了CHECKPOINT轮的网络存档进行测试

import tensorflow.compat.v1 as tf

import os
import math
import time
import random
import numpy as np
from collections import deque
from queue import deque
from GomokuBoard import GomokuBoard
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.disable_v2_behavior()

class DQN():
    def __init__(self):
        tf.reset_default_graph()
        self.SIZE = GomokuBoard.SIZE
        self.state_dim = self.SIZE * self.SIZE
        self.action_dim = self.SIZE * self.SIZE
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_E

        self.create_Q()
        self.create_targetQ()
        self.targetQ_step = Q_STEP
        self.train_method()
        # 输出是否支持cuda
        print("is_gpu_available: ", tf.test.is_gpu_available())
        print("is_built_with_cuda: ", tf.test.is_built_with_cuda())

        # GPU设置
        gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.0, allow_growth=True)
        config = tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=config)

        # 网络初始化
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        # save checkpoints
        self.saver = tf.train.Saver()
    
    def save_checkpoint(self, directory, step):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, 'model.ckpt'), global_step=step)

    def load_checkpoint(self, directory):
        print(f'direc:{directory}')
        self.saver.restore(self.sess, directory)

    def create_Q(self):
        # 网络权值
        W1 = self.weight_variable([5, 5, 1, 16])
        b1 = self.bias_variable([16])  # 5*5*16
        W2 = self.weight_variable([5*5*16+1, 225])
        b2 = self.bias_variable([1, 225])

        # 输入层
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        self.turn = tf.placeholder("float", [None, 1])

        y0 = tf.reshape(self.state_input, [-1, 15, 15, 1])
        # 卷积层
        h1 = tf.nn.relu(self.conv2d(y0, W1) + b1)
        y1 = self.max_pool_3_3(h1)  # 5*5*16
        # 全连接层
        h2 = tf.concat([tf.reshape(y1, [-1, 5 * 5 * 16]), self.turn], 1)
        self.Q_value = tf.matmul(h2, W2)+b2
        # 保存权重
        self.Q_weights = [W1, b1, W2, b2]

    def create_targetQ(self):
        # 网络权值
        W1 = self.weight_variable([5, 5, 1, 16])
        b1 = self.bias_variable([16])  # 5*5*16
        W2 = self.weight_variable([5*5*16+1, 225])
        b2 = self.bias_variable([1, 225])

        # 输入层
        # self.state_input = tf.placeholder("float", [None, self.state_dim])
        # self.turn = tf.placeholder("float", [None, 1])

        y0 = tf.reshape(self.state_input, [-1, 15, 15, 1])
        # 卷积层
        h1 = tf.nn.relu(self.conv2d(y0, W1) + b1)
        y1 = self.max_pool_3_3(h1)  # 5*5*16

        # 全连接层
        h2 = tf.concat([tf.reshape(y1, [-1, 5 * 5 * 16]), self.turn], 1)
        self.targetQ_value = tf.matmul(h2, W2)+b2
        # 保存权重
        self.targetQ_weights = [W1, b1, W2, b2]

    def copy(self):
        for i in range(len(self.Q_weights)):
            self.sess.run(
                tf.assign(self.targetQ_weights[i], self.Q_weights[i]))

    def train_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(tf.multiply(
            self.Q_value, self.action_input), reduction_indices=1)  # mul->matmul
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.train = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])
        if len(self.replay_buffer) > REPLAY_NUM:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_NUM:
            self.train_Q_network()

    def modify_last_reward(self, new_reward):
        v = self.replay_buffer.pop()
        v[2] = new_reward
        self.replay_buffer.append(v)

    def train_Q_network(self):
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer, BATCH_NUM)
        state_batch = [data[0][0] for data in minibatch]
        state_batch_turn = [data[0][1] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3][0] for data in minibatch]
        next_state_batch_turn = [data[3][1] for data in minibatch]
        y_batch = []

        Q_value_batch = self.sess.run(self.targetQ_value, feed_dict={
            self.state_input: next_state_batch, self.turn: next_state_batch_turn})

        for i in range(0, BATCH_NUM):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.max(Q_value_batch[i]))
        self.sess.run(self.train, feed_dict={
                      self.y_input: y_batch, self.action_input: action_batch, self.state_input: state_batch, self.turn: state_batch_turn})

        if self.time_step % self.targetQ_step == 0:
            self.copy()


    def egreedy_action(self, state):
        Q_value = self.sess.run(self.Q_value, feed_dict={
                                self.state_input: [state[0]], self.turn: [state[1]]})[0]

        min_v = Q_value[np.argmin(Q_value)] - 1  # 最小的Q_value -1
        valid_action = []

        for i in range(len(Q_value)):  # 遍历每一个落子点
            if state[0][i] == EMPTY:
                valid_action.append(i)
            else:
                Q_value[i] = min_v

        # 以epsilon的概率随机落子
        if random.random() <= self.epsilon:
            l = len(valid_action)
            if l == 0:
                return -1
            else:
                return valid_action[random.randint(0, len(valid_action) - 1)]
        else:  # 其它清空，选取Q最大的点落子
            return np.argmax(Q_value)

    def action(self, state):
       # 计算当前局面的所有Q值
        Q_value = self.sess.run(self.Q_value, feed_dict={
            self.state_input: [state[0]], self.turn: [state[1]]})[0]

        min_v = Q_value[np.argmin(Q_value)] - 1  # 最小的Q_value -1
        valid_action = []

        for i in range(len(Q_value)):  # 遍历每一个落子点
            if state[0][i] == EMPTY:
                valid_action.append(i)
            else:
                Q_value[i] = min_v
        return np.argmax(Q_value)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_3_3(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

# 自弈测试
def test():
    total_reward = 0
    chess = GomokuBoard()
    chess.reset()
    
    state = chess.board
    camp = np.zeros([1])
    camp[0] = -1
    state = np.reshape(state, [-1])
    state = [state, camp]

    # load checkpoints
    checkpoint_dir = './checkpoints'
    checkpoint_step = 'model.ckpt-' + str(CHECKPOINT)

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_step)
    agent.load_checkpoint(checkpoint_path)
    
    for j in range(225):
        action = agent.action(state)  # 按照Q网络的走一步
        action = [math.floor(action/GomokuBoard.SIZE), action %
                  GomokuBoard.SIZE, camp]  # 转化为二维棋盘坐标

        state, reward, done, _ = chess.draw_XY(action[0], action[1])
        state = np.reshape(state, [-1])
        
        if j % 2 == 0:
            camp[0] = 1
        else:
            camp[0] = -1
        state = [state, camp]
        total_reward += reward

        # 打印棋盘
        os.system("cls")
        chess.printChess()
        if camp == 1:
            print(" BLACK %d   %d\n" % (action[0], action[1]))
        else:
            print(" WHITE %d   %d\n" % (action[0], action[1]))
        time.sleep(2)

        # 结束判断
        if done:
            print('done')
            time.sleep(5)
            break

# Colab运行时随时存储到云端硬盘
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir('/content/drive/My Drive/')

train_step = []
agent = DQN()  # 创建网络对象
agent.copy()   # 复制Q网络参数到targetQ

def main():
    # 超参数
    EPISODE = 100000
    STEP = 300
    save_freq = 1000

    chess = GomokuBoard()  # 创建一个主棋盘

    for episode in tqdm(range(EPISODE), mininterval = 30):
        chess.reset()
        state = chess.board
        camp = np.zeros([1])
        camp[0] = -1
        state = np.reshape(state, [-1])
        state = [state, camp]
        
        if episode%100==99 and agent.epsilon>FINAL_E:
            agent.epsilon *= 0.99  # 每100局，减小随机选择落子点的概率
            
        # 训练
        for step in range(STEP):
            # 自己下一步棋
            action_1d = agent.egreedy_action(state)  # 有随机概率的走一步
            action_2d = [math.floor(action_1d / GomokuBoard.SIZE), action_1d %
                         GomokuBoard.SIZE, camp]  # 转化为二维棋盘坐标
            # 在模拟棋盘上落子
            next_state_2d, reward, done, _ = chess.draw_XY(
                action_2d[0], action_2d[1])
            next_state = np.reshape(next_state_2d, [-1])
            # 构造数据
            if step % 2 == 0:
                camp[0] = 1
            else:
                camp[0] = -1
            next_state = [next_state, camp]
            # 丢入经验池  执行训练
            agent.perceive(state, action_1d, reward, next_state, done)
            state = next_state
            if done:
                train_step.append(step)
                # print("done step:%d  episode:%d epsilon:%.5f" % (step, episode,agent.epsilon))
                break  
        # 在适当的地方调用 save_checkpoint 方法保存模型
        if (episode+1)%save_freq==0:
            agent.save_checkpoint('checkpoints', episode+1)

def train():
    time_start = time.time()
    main()
    time_end = time.time()
    print('totally cost', time_end - time_start)

import sys
if __name__ == '__main__':
    act = 'test' # sys.argv[1]
    if IS_TRAIN:
        act = 'train'
    if act == 'train':
        train()
    else:
        test()