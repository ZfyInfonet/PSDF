import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)
tf.random.set_seed(1)


class EvalNet(tf.keras.Model):

    def __init__(self, n_actions):
        super().__init__()
        self.layer1 = layers.Dense(50, activation='relu')
        self.layer2 = layers.Dense(50, activation='relu')
        self.layer3 = layers.Dense(50, activation='relu')
        self.layer4 = layers.Dense(n_actions, activation=None)
        # self.dropout = layers.Dropout(0.5)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


"""
class TargetNet(tf.keras.Model):

    def __init__(self, n_actions):
        super().__init__()
        self.layer1 = layers.Dense(30, activation='relu')
        self.layer2 = layers.Dense(15, activation='relu')
        self.layer3 = layers.Dense(n_actions, activation=None)
        # self.dropout = layers.Dropout(0.5)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer3
"""


class DeepQNetwork:

    def __init__(
            self,
            n_actions,
            n_features,
            eval_model,
            target_model,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None
            # output_graph=False
    ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.eval_model = eval_model
        self.target_model = target_model

        self.eval_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        self.cost_his = []
        self.memory_counter = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_model.predict(observation)
            # print("action_value", actions_value)
            # print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def choose_action_for_test(self, observation):
        observation = observation[np.newaxis, :]
        # forward feed the observation and get q value for every actions
        actions_value = self.eval_model.predict(observation)
        print("action_value", actions_value)
        # print(actions_value)
        action = np.argmax(actions_value)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # self.eval_model.save_weights('./DQN_model/eval_model')
            # self.target_model.load_weights('./DQN_model/eval_model')
            # for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
            #    target_layer.set_weights(eval_layer.get_weights())
            self.target_model.set_weights(self.eval_model.get_weights())
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.target_model.predict(batch_memory[:, -self.n_features:])
        q_eval = self.eval_model.predict(batch_memory[:, :self.n_features])

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        cost = self.eval_model.train_on_batch(batch_memory[:, :self.n_features], q_target)
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
