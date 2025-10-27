import torch
import pygame
import numpy as np
from car_env import CarEnvironment

class BCTester:
    def __init__(self, env, model, model_type='state'):
        self.env = env
        self.model = model
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def test_episode(self, max_steps=1000, render=True):
        """测试一个episode"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        actions_taken = {0: 0, 1: 0, 2: 0}  # 统计动作分布
        
        print("开始测试...")
        print("按Q退出测试")
        
        running = True
        while running and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
            
            if not running:
                break
            
            # 使用模型预测动作
            with torch.no_grad():
                if self.model_type == 'state':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_probs = self.model(state_tensor)
                elif self.model_type == 'pixel':
                    pixel_state = self.env.get_pixel_state()
                    pixel_tensor = torch.FloatTensor(pixel_state).unsqueeze(0).unsqueeze(0).to(self.device)
                    action_probs = self.model(pixel_tensor)
                elif self.model_type == 'ensemble':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    pixel_state = self.env.get_pixel_state()
                    pixel_tensor = torch.FloatTensor(pixel_state).unsqueeze(0).unsqueeze(0).to(self.device)
                    action_probs = self.model(state_tensor, pixel_tensor)
                
                action = torch.argmax(action_probs, dim=1).item()
                actions_taken[action] += 1
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            
            if render:
                self.env.render()
                self.env.clock.tick(60)
            
            if done:
                print(f"Episode 结束! 步数: {steps}, 奖励: {total_reward:.2f}")
                break
        
        # 打印统计信息
        print(f"\n测试结果:")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward:.2f}")
        print(f"动作分布: 左转{actions_taken[0]}, 直行{actions_taken[1]}, 右转{actions_taken[2]}")
        
        return total_reward, steps, actions_taken
    
    def test_multiple_episodes(self, num_episodes=5):
        """测试多个episode"""
        results = []
        
        for episode in range(num_episodes):
            print(f"\n第 {episode + 1}/{num_episodes} 个测试episode")
            reward, steps, actions = self.test_episode()
            results.append({
                'episode': episode + 1,
                'reward': reward,
                'steps': steps,
                'actions': actions
            })
            
            if episode < num_episodes - 1:
                input("按回车继续下一个episode...")
        
        # 打印总体统计
        print("\n" + "="*50)
        print("总体测试结果:")
        avg_reward = np.mean([r['reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均步数: {avg_steps:.2f}")
        
        return results