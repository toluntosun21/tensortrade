import random
import numpy as np
import tensorflow as tf
import copy
from tensortrade.data import DataFeed

from collections import namedtuple

from tensortrade.agents import Agent, ReplayMemory

A2C_LSTM_Transition = namedtuple('A2CTransition', ['state', 'action', 'reward', 'done', 'value'])


class A2C_LSTM_Agent(Agent):

    def __init__(self,
                 env: 'TradingEnvironment',
                 test_env: 'TradingEnvironment' = None,
                 shared_network: tf.keras.Model = None,
                 actor_network: tf.keras.Model = None,
                 critic_network: tf.keras.Model = None):
        self.env = env
        self.test_env = test_env or copy.deepcopy(env)
        self.n_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape

        self.shared_network = shared_network or self._build_shared_network()
        self.actor_network = actor_network or self._build_actor_network()
        self.critic_network = critic_network or self._build_critic_network()

        self.env.agent_id = self.id

    def _build_shared_network(self):
        self.LSTM = tf.keras.layers.LSTM(50, return_sequences=True, stateful=True)

        network = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.InputLayer(input_shape=self.observation_shape)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=2)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
            self.LSTM

        ])
        return network

    def _build_actor_network(self):
        actor_head = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(self.n_actions, activation='relu')
        ])
        return tf.keras.Sequential([self.shared_network, actor_head])

    def _build_critic_network(self):
        critic_head = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        return tf.keras.Sequential([self.shared_network, critic_head])

    def restore(self, path: str, **kwargs):
        actor_filename: str = kwargs.get('actor_filename', None)
        critic_filename: str = kwargs.get('critic_filename', None)

        if not actor_filename or not critic_filename:
            raise ValueError(
                'The `restore` method requires a directory `path`, a `critic_filename`, and an `actor_filename`.')

        self.actor_network = tf.keras.models.load_model(path + actor_filename)
        self.critic_network = tf.keras.models.load_model(path + critic_filename)

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            suffix = self.id + "__" + str(episode).zfill(3) + ".hdf5"
            actor_filename = "actor_network__" + suffix
            critic_filename = "critic_network__" + suffix
        else:
            actor_filename = "actor_network__" + self.id + ".hdf5"
            critic_filename = "critic_network__" + self.id + ".hdf5"

        self.actor_network.save(path + actor_filename)
        self.critic_network.save(path + critic_filename)

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        threshold: float = kwargs.get('threshold', 0)

        rand = random.random()

        if rand < threshold:
            return np.random.choice(self.n_actions)
        else:
            logits = self.actor_network(state[None, None, :], training=False)
            return tf.squeeze(tf.squeeze(tf.random.categorical(logits[0], 1), axis=-1), axis=-1)

    def _apply_gradient_descent(self,
                                memory: ReplayMemory,
                                batch_size: int,
                                learning_rate: float,
                                discount_factor: float,
                                entropy_c: float,):

        if hasattr(self, 'trained_lstm_states'):
            self.LSTM.states = copy.deepcopy(self.trained_lstm_states)
        else:
            self.trained_lstm_states = copy.deepcopy(self.LSTM.states)
            self.LSTM.reset_states()

        huber_loss = tf.keras.losses.Huber()
        wsce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        transitions = memory.tail(batch_size)
        batch = A2C_LSTM_Transition(*zip(*transitions))

        states = tf.convert_to_tensor(batch.state)
        actions = tf.convert_to_tensor(batch.action)
        rewards = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
        dones = tf.convert_to_tensor(batch.done)
        values = tf.convert_to_tensor(batch.value)

        returns = []
        exp_weighted_return = 0

        for reward, done in zip(rewards[::-1], dones[::-1]):
            exp_weighted_return = reward + discount_factor * exp_weighted_return * (1 - int(done))
            returns += [exp_weighted_return]

        returns = returns[::-1]

        with tf.GradientTape() as tape:
            state_values = self.critic_network(states[None,:])
            critic_loss_value = huber_loss(returns, state_values)

        gradients = tape.gradient(critic_loss_value, self.critic_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.critic_network.trainable_variables))

        with tf.GradientTape() as tape:
            returns = tf.reshape(returns, [batch_size, 1])
            advantages = returns - values

            actions = tf.cast(actions, tf.int32)
            logits = self.actor_network(states[None,:])
            policy_loss_value = wsce_loss(actions, logits, sample_weight=advantages)

            probs = tf.nn.softmax(logits)
            entropy_loss_value = tf.keras.losses.categorical_crossentropy(probs, probs)
            policy_total_loss_value = policy_loss_value - entropy_c * entropy_loss_value

        gradients = tape.gradient(policy_total_loss_value,
                                  self.actor_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.actor_network.trainable_variables))
        self.trained_lstm_states = copy.deepcopy(self.LSTM.states)

    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 400)
        entropy_c: int = kwargs.get('entropy_c', 0.0001)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)
        train_end: float = kwargs.get('train_end', 0.3)

        memory = ReplayMemory(memory_capacity, transition_type=A2C_LSTM_Transition)
        episode = 0
        steps_done = 0
        total_reward = 0
        stop_training = False

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('====      AGENT ID: {}      ===='.format(self.id))

        while episode < n_episodes and not stop_training:
            if not episode or not self.env.feed.has_next():
                state = self.env.reset()
            done = False
            steps_done = 0
            if episode:
                #self.LSTM.reset_states()
                memory = ReplayMemory(memory_capacity, transition_type=A2C_LSTM_Transition)


            self.env.portfolio.reset()
            print(self.env.portfolio.balances)
            print('====      TRAIN EPISODE ID ({}/{}): {}      ===='.format(episode + 1,
                                                                      n_episodes,
                                                                      self.env.episode_id))

            while not done:
                if steps_done % 24 == 0: #each day
                    print("step {}/{}".format(steps_done, n_steps))
                    print(self.env.portfolio.balances)
                    print(self.env.portfolio.net_worth)
                    print('exchange:', state[-1][0])

                if not self.env.feed.has_next():
                    done = True
                    continue
                threshold = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)
                value = self.critic_network(state[None, None, :], training=False)
                value = tf.squeeze(value, axis=-1)

                memory.push(state, action, reward, done, value)

                state = next_state
                total_reward += reward
                steps_done += 1

                if self.env.portfolio.net_worth < self.env.portfolio.initial_net_worth * train_end:
                    done = True
                    continue

                if len(memory) < batch_size:
                    continue

                if True or steps_done % batch_size == 0:
                    self._apply_gradient_descent(memory,
                                             batch_size,
                                             learning_rate,
                                             discount_factor,
                                             entropy_c)

                if n_steps and steps_done >= n_steps:
                    done = True
                    #stop_training = True

            # VALIDATION

            test_state = self.test_env.reset()
            self.LSTM.reset_states()
            done = False
            steps_done = 0
            threshold = 0

            print(self.env.portfolio.balances)
            print('====      TEST EPISODE ID ({}/{}): {}      ===='.format(episode + 1,
                                                                      n_episodes,
                                                                      self.test_env.episode_id))

            while not done:
                if steps_done % 24 == 0: #each day
                    print("step {}/{}".format(steps_done, n_steps))
                    print(self.test_env.portfolio.balances)
                    print(self.test_env.portfolio.net_worth)
                    print('exchange:', test_state[-1][0])

                if not self.test_env.feed.has_next():
                    done = True
                    continue

                action = self.get_action(test_state, threshold=threshold)
                next_state, reward, done, _ = self.test_env.step(action)
                value = self.critic_network(test_state[None, None, :], training=False)
                value = tf.squeeze(value, axis=-1)
                test_state = next_state
                steps_done += 1
                if self.test_env.portfolio.net_worth < self.test_env.portfolio.initial_net_worth * train_end:
                    done = True
                    continue

            portfolio_perf = self.test_env.portfolio.performance.values
            np.savetxt(save_path+'/test{}.csv'.format(episode+1), portfolio_perf, delimiter=',', fmt='%s')


            self.LSTM.reset_states()

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes):
                self.save(save_path, episode=episode)

            episode += 1

        mean_reward = total_reward / steps_done

        return mean_reward