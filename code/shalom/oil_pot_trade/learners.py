import os
import logging
import abc
import collections
import threading
import time
import json
import numpy as np
import math
from tqdm import tqdm

from shalom.oil_pot_trade.environment import Environment
from shalom.oil_pot_trade.agent import Agent
from shalom.oil_pot_trade.networks import Network, LSTMNetwork
from shalom.oil_pot_trade.visualizer import Visualizer
from shalom.oil_pot_trade import utils
from shalom.oil_pot_trade import settings

logger = logging.getLogger(settings.LOGGER_NAME)


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None, 
                chart_data=None, training_data=None,
                min_trading_price=100000, max_trading_price=10000000, 
                net='dnn', num_steps=1, lr=0.0005, 
                discount_factor=0.9, num_epoches=1000,
                balance=100000000, start_epsilon=1,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True, gen_output=True):
        # 인자 확인
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_loss = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        # 초기 학습률과 최종 학습률
        self.initial_lr = lr
        #self.final_lr = 0.00001  # final_lr을 0.00001로 설정 (고정)

        # 에포크 수와 입력된 epoch 값을 받아서 decay_rate 계산
        self.num_epoches = num_epoches
        #self.calculate_decay_rate(self.num_epoches)        
        # 로그 등 출력 경로
        self.output_path = output_path
        self.gen_output = gen_output

    #def calculate_decay_rate(self, num_epoches):
        # 에포크 수와 final_lr을 이용하여 decay_rate 계산
        #self.decay_rate = lr_decay = (self.initial_lr - self.final_lr) / num_epoches

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        self.value_network = LSTMNetwork(
            input_dim=self.num_features, 
            output_dim=self.agent.NUM_ACTIONS, 
            lr=self.lr, num_steps=self.num_steps, 
            shared_network=shared_network,
            activation=activation, loss=loss)

        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation='sigmoid', 
                            loss='binary_crossentropy'):
        self.policy_network = LSTMNetwork(
            input_dim=self.num_features, 
            output_dim=self.agent.NUM_ACTIONS, 
            lr=self.lr, num_steps=self.num_steps, 
            shared_network=shared_network,
            activation=activation, loss=loss)

        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        #self.memory_loss = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self):
        pass

    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            self.loss = loss
        return self.loss


    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,lr=self.lr,
            loss=self.memory_loss,
            reward=self.memory_reward
        )
        if 0:
            self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{self.stock_code}.png'))
        else:
            folder_path = os.path.join(self.epoch_summary_dir, self.stock_code)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            self.visualizer.save(os.path.join(self.epoch_summary_dir, self.stock_code, f'epoch_summary_{epoch_str}.png'))
            self.visualizer.close()

    def early_stopping(self, count, threshold=50, min_epoch=100):
        """
        Early stopping based on the loss.
        Args:
            threshold (int): Number of epochs to wait before stopping if the loss doesn't decrease.
            min_epoch (int): Minimum number of epochs to run before considering stopping.
        Returns:
            bool: True if the training should be stopped, False otherwise.
        """
        last_losses = self.memory_loss[-threshold:]

        if count <= min_epoch:
            return False

        if all(last_losses[i] >= last_losses[i+1] for i in range(threshold - 1)):
            print("Early stopping")
            return True
        return False

    def run(self, learning=True):
        info = (
            f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net} DF:{self.discount_factor}'
        )
        with self.lock:
            logger.debug(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비 
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        #if self.gen_output:
            #self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
        self.epoch_summary_dir = self.output_path
        #    if not os.path.isdir(self.epoch_summary_dir):
        #        os.makedirs(self.epoch_summary_dir)
        #    else:
        #        for f in os.listdir(self.epoch_summary_dir):
        #            os.remove(os.path.join(self.epoch_summary_dir, f))
        if os.path.exists(os.path.join(self.epoch_summary_dir, f'epoch_summary_{self.stock_code}.png')):
            os.remove(os.path.join(self.epoch_summary_dir, f'epoch_summary_{self.stock_code}.png'))  #add remove
        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0
        self.memory_loss = []
        # 에포크 반복
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()
            # 학습률 업데이트
            #self.lr -= self.decay_rate  # 학습률을 갱신
            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
            else:
                epsilon = self.start_epsilon

            for i in tqdm(range(len(self.training_data)), leave=False):
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 보상 획득
                reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

            # 에포크 종료 후 학습
            if learning:
                self.memory_loss.append(self.fit())

            if True == self.early_stopping(epoch): break
            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(self.num_epoches))
            #epoch_str = self.stock_code#str(epoch + 1).rjust(num_epoches_digit, '0')
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epoches}] '
                f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')

            # 에포크 관련 정보 가시화
            if self.gen_output:
                if self.num_epoches == 1 or (epoch + 1) % max(int(self.num_epoches / 100), 1) == 0:
                    self.visualize(epoch_str, self.num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1



        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f} '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    def predict(self):
        # 에이전트 초기화
        self.agent.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample))
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample))
            
            # 신경망에 의한 행동 결정
            action, confidence, _ = self.agent.decide_action(pred_value, pred_policy, 0)
            result.append((self.environment.observation.iloc[0], int(action), float(confidence)))

        #if self.gen_output:
        #    with open(os.path.join(self.output_path, f'pred_{self.stock_code}.json'), 'w') as f:
        #        print(json.dumps(result), file=f)
        return result

    def valid(self, data):
        """
        Validate the trained model.
        Args:
            data (DataFrame): Validation data.
        Returns:
            float: The average reward obtained during the validation.
        """
        # Reset the agent and other variables.
        self.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        rewards = []

        # Iterate over the validation data.
        for _ in tqdm(range(len(data)), leave=False):
            # Create the sample.
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # Append the sample to the queue.
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # Predict using the value and policy networks.
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample))
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample))

            # Decide the action using the networks.
            action, confidence, _ = self.agent.decide_action(pred_value, pred_policy, 0)

            # Act and get the reward.
            reward = self.agent.act(action, confidence)
            rewards.append(reward)

        # Calculate and return the average reward.
        avg_reward = np.mean(rewards)
        return avg_reward

    def test(self, data):
        """
        Test the trained model.
        Args:
            data (DataFrame): Test data.
        Returns:
            float: The average reward obtained during the test.
        """
        # Reset the agent and other variables.
        self.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        rewards = []

        # Iterate over the test data.
        for _ in tqdm(range(len(data)), leave=False):
            # Create the sample.
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # Append the sample to the queue.
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # Predict using the value and policy networks.
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample))
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample))

            # Decide the action using the networks.
            action, confidence, _ = self.agent.decide_action(pred_value, pred_policy, 0)

            # Act and get the reward.
            reward = self.agent.act(action, confidence)
            rewards.append(reward)

        # Calculate and return the average reward.
        avg_reward = np.mean(rewards)
        return avg_reward


class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None, 
        value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps, 
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i, :] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(r)
            value_max_next = value.max()
        return x, y_value, y_policy

class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        #mem_cnt = len(self.memory_reward)

        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample

            #r = (reward_next - reward) * 2
            r = (self.memory_reward[-1] - reward)/(i+1) + (reward_next - reward)
            
            reward_next = reward

            y_value[i, :] = value
            y_value[i, action] = np.tanh(r + self.discount_factor * value_max_next)

            advantage = y_value[i, action] - y_value[i].mean()
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(advantage)
            value_max_next = value.max()
        return x, y_value, y_policy
