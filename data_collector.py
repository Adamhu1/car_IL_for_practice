import pygame
import numpy as np
import pickle
import os
from datetime import datetime
from car_env import CarEnvironment

class DataCollector:
    def __init__(self, env):
        self.env = env
        self.dataset = {
            'states': [],
            'pixel_states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'timestamps': []
        }
        self.episode_data = []
        
    def collect_episode(self, max_steps=1000):
        """收集一个episode的数据"""
        state = self.env.reset()
        episode_reward = 0
        steps = 0
        
        print("开始收集数据...")
        print("控制说明: A-左转, D-右转, 其他键-直行")
        print("按Q退出数据收集")
        
        running = True
        while running and steps < max_steps:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
            
            if not running:
                break
            
            # 获取键盘输入
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:  # A键左转
                action = 0
            elif keys[pygame.K_d]:  # D键右转
                action = 2
            else:  # 直行
                action = 1
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(action)
            pixel_state = self.env.get_pixel_state()
            
            # 保存数据
            self.episode_data.append({
                'state': state.copy(),
                'pixel_state': pixel_state.copy(),
                'action': action,
                'reward': reward,
                'done': done,
                'timestamp': datetime.now()
            })
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            self.env.render()
            
            if done:
                print(f"Episode 结束! 步数: {steps}, 奖励: {episode_reward:.2f}")
                break
        
        return episode_reward, steps
    
    def save_episode_data(self):
        """保存当前episode的数据到数据集"""
        for data_point in self.episode_data:
            self.dataset['states'].append(data_point['state'])
            self.dataset['pixel_states'].append(data_point['pixel_state'])
            self.dataset['actions'].append(data_point['action'])
            self.dataset['rewards'].append(data_point['reward'])
            self.dataset['dones'].append(data_point['done'])
            self.dataset['timestamps'].append(data_point['timestamp'])
        
        self.episode_data = []
        print(f"数据集大小: {len(self.dataset['states'])}")
    
    def collect_multiple_episodes(self, num_episodes=5):
        """收集多个episode的数据"""
        for episode in range(num_episodes):
            print(f"\n开始第 {episode + 1}/{num_episodes} 个episode")
            reward, steps = self.collect_episode()
            self.save_episode_data()
            
            if episode < num_episodes - 1:
                cont = input("继续下一个episode? (y/n): ")
                if cont.lower() != 'y':
                    break
    
    def save_dataset(self, filename=None):
        """保存数据集到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expert_data_{timestamp}.pkl"
        
        # 创建data目录
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.dataset, f)
        
        print(f"数据集已保存到: {filepath}")
        print(f"总样本数: {len(self.dataset['states'])}")
        
        return filepath
    
    def load_dataset(self, filename):
        """从文件加载数据集"""
        filepath = os.path.join('data', filename)
        with open(filepath, 'rb') as f:
            self.dataset = pickle.load(f)
        print(f"已加载数据集: {filepath}")
        print(f"样本数: {len(self.dataset['states'])}")
    
    def get_dataset_stats(self):
        """获取数据集统计信息"""
        if not self.dataset['actions']:
            return "数据集为空"
        
        actions = np.array(self.dataset['actions'])
        rewards = np.array(self.dataset['rewards'])
        
        stats = {
            'total_samples': len(actions),
            'action_distribution': {
                'left': np.sum(actions == 0),
                'straight': np.sum(actions == 1),
                'right': np.sum(actions == 2)
            },
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            }
        }
        
        return stats