import pygame
import numpy as np
import cv2
import random  # 新增
from typing import Tuple, Dict, Any, Optional

class CarEnvironment:
    def __init__(self, render_mode="human",
                 randomize: bool = True,        # 重置时是否随机
                 seed: Optional[int] = None,
                 road_width_range=(260, 520),   # 道路宽度范围
                 road_y_margin=120,             # 道路中心线距离上下边界的最小边距
                 start_x_offset=(40, 120),      # 起点距离左侧车道边的偏移范围
                 start_y_jitter=(-60, 60),      # 起点在道路中心线的上下抖动范围
                 start_angle_range=(-25, 25),   # 起始朝向角度范围（度）
                 obstacle_count_range=(1, 4)    # 障碍物数量范围
                 ):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Imitation Learning - Self Driving Car")
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode

        # 随机化配置
        self.randomize = randomize
        self.road_width_range = road_width_range
        self.road_y_margin = road_y_margin
        self.start_x_offset = start_x_offset
        self.start_y_jitter = start_y_jitter
        self.start_angle_range = start_angle_range
        self.obstacle_count_range = obstacle_count_range
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 汽车参数
        self.car_x, self.car_y = 100, 300
        self.car_width, self.car_height = 40, 20
        self.car_speed = 1
        self.car_angle = 0  # 角度，0表示向右
        
        # 道路参数
        self.road_width = 400
        self.road_y = 300
        self.road_left = 200
        self.road_right = 600
        
        # 轨迹记录
        self.trajectory = []
        
        # 障碍物
        self.obstacles = [
            {'x': 400, 'y': 280, 'width': 30, 'height': 40},
            {'x': 500, 'y': 320, 'width': 30, 'height': 40}
        ]
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        if self.randomize:
            # 随机道路宽度与中心线位置
            self.road_width = int(random.randint(*self.road_width_range))
            self.road_y = int(random.randint(self.road_y_margin, self.height - self.road_y_margin))
            self.road_left = (self.width - self.road_width) // 2
            self.road_right = self.road_left + self.road_width
        else:
            # 使用固定道路参数（与 __init__ 中一致）
            self.road_width = 400
            self.road_y = 300
            self.road_left = 200
            self.road_right = 600

        # 随机起点姿态（位于车道左侧附近，朝向近似水平）
        left_safe = self.road_left + max(self.start_x_offset[0], 10)
        right_safe = self.road_left + min(self.start_x_offset[1], self.road_width - 40)
        self.car_x = float(random.randint(left_safe, right_safe))
        self.car_y = float(self.road_y + random.randint(*self.start_y_jitter))
        self.car_angle = float(random.uniform(*self.start_angle_range))
        self.trajectory = []

        # 障碍物数量与位置
        count = random.randint(*self.obstacle_count_range) if self.randomize else len(getattr(self, "obstacles", [])) or 2
        self.obstacles = self._random_obstacles(count)

        return self.get_state()
    
    def _random_obstacles(self, count: int):
        """在道路范围内随机生成障碍，避开起点附近"""
        obstacles = []
        tries_per_obs = 20
        for _ in range(count):
            placed = False
            for _ in range(tries_per_obs):
                w = random.randint(24, 40)
                h = random.randint(24, 40)
                x = random.uniform(self.road_left + 20, self.road_right - 20 - w)
                y = random.uniform(60, self.height - 60)

                # 避开起点
                if abs(x - self.car_x) < 80 and abs(y - self.car_y) < 80:
                    continue
                obstacles.append({'x': x, 'y': y, 'width': w, 'height': h})
                placed = True
                break
            if not placed:
                # 兜底：若多次尝试失败，放到车道中部
                obstacles.append({'x': (self.road_left + self.road_right) / 2,
                                  'y': self.road_y + random.randint(-100, 100),
                                  'width': 30, 'height': 30})
        return obstacles

    def get_state(self) -> np.ndarray:
        """获取状态向量"""
        # 计算到道路边界的距离
        dist_to_left = self.car_x - self.road_left
        dist_to_right = self.road_right - self.car_x
        dist_to_center = abs(self.car_y - self.road_y)
        
        # 计算到最近障碍物的距离
        min_obstacle_dist = float('inf')
        for obstacle in self.obstacles:
            dx = obstacle['x'] - self.car_x
            dy = obstacle['y'] - self.car_y
            dist = np.sqrt(dx**2 + dy**2)
            min_obstacle_dist = min(min_obstacle_dist, dist)
        
        state = np.array([
            self.car_x / self.width,                    # 归一化x坐标
            self.car_y / self.height,                   # 归一化y坐标
            dist_to_left / self.road_width,            # 到左边界的距离
            dist_to_right / self.road_width,           # 到右边界的距离
            dist_to_center / 100,                      # 到道路中心的距离
            min_obstacle_dist / 500,                   # 到最近障碍物的距离
            self.car_angle / 360                       # 汽车角度
        ], dtype=np.float32)
        
        return state
    
    def get_pixel_state(self) -> np.ndarray:
        """获取像素状态（用于图像输入）"""
        # 创建一个小的渲染表面
        surface = pygame.Surface((84, 84))
        surface.fill((255, 255, 255))
        
        # 绘制道路
        road_rect = pygame.Rect(
            int((self.road_left - self.car_x + 42) * 0.5),
            int((self.road_y - 50 - self.car_y + 42) * 0.5),
            int(self.road_width * 0.5),
            100
        )
        pygame.draw.rect(surface, (100, 100, 100), road_rect)
        
        # 绘制汽车（在中心）
        car_rect = pygame.Rect(40, 40, 4, 2)
        pygame.draw.rect(surface, (255, 0, 0), car_rect)
        
        # 转换为numpy数组
        pixel_array = pygame.surfarray.array3d(surface)
        pixel_array = np.transpose(pixel_array, (1, 0, 2))  # 转置为(H, W, C)
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)  # 转为灰度
        
        return pixel_array / 255.0  # 归一化
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        # 动作: 0-左转, 1-保持直行, 2-右转
        if action == 0:  # 左转
            self.car_angle = (self.car_angle - 5) % 360
        elif action == 2:  # 右转
            self.car_angle = (self.car_angle + 5) % 360
        
        # 根据角度移动
        rad = np.radians(self.car_angle)
        self.car_x += self.car_speed * np.cos(rad)
        self.car_y += self.car_speed * np.sin(rad)
        
        # 限制在屏幕内
        self.car_x = max(0, min(self.car_x, self.width))
        self.car_y = max(0, min(self.car_y, self.height))
        
        # 记录轨迹
        self.trajectory.append((self.car_x, self.car_y))
        if len(self.trajectory) > 100:  # 只保留最近100个点
            self.trajectory.pop(0)
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否结束
        done = self._check_done()
        
        return self.get_state(), reward, done, {}
    
    def _calculate_reward(self) -> float:
        """计算奖励"""
        reward = 0.1  # 生存奖励
        
        # 保持在道路内的奖励
        if self.road_left < self.car_x < self.road_right:
            reward += 0.5
        else:
            reward -= 0.3
        
        # 保持在道路中心线的奖励
        center_dist = abs(self.car_y - self.road_y)
        if center_dist < 20:
            reward += 0.2
        
        # 避免障碍物的惩罚
        for obstacle in self.obstacles:
            if (abs(self.car_x - obstacle['x']) < 30 and 
                abs(self.car_y - obstacle['y']) < 30):
                reward -= 1.0
        
        # 前进的奖励
        reward += (self.car_x - 100) / 1000
        
        return reward
    
    def _check_done(self) -> bool:
        """检查是否结束"""
        # 检查是否出界
        if (self.car_x < 50 or self.car_x > self.width - 50 or 
            self.car_y < 50 or self.car_y > self.height - 50):
            return True
        
        # 检查碰撞障碍物
        for obstacle in self.obstacles:
            if (abs(self.car_x - obstacle['x']) < 25 and 
                abs(self.car_y - obstacle['y']) < 25):
                return True
        
        # 最大步数
        if len(self.trajectory) > 1000:
            return True
        
        return False
    
    def render(self):
        """渲染环境"""
        pygame.event.pump()
        self.screen.fill((255, 255, 255))  # 白色背景
        
        # 绘制道路
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (self.road_left, self.road_y - 50, 
                         self.road_width, 100))
        
        # 绘制道路中心线
        for i in range(self.road_left, self.road_right, 40):
            pygame.draw.rect(self.screen, (255, 255, 0), 
                           (i, self.road_y - 2, 20, 4))
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 255), 
                           (obstacle['x'], obstacle['y'], 
                            obstacle['width'], obstacle['height']))
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.screen, (0, 255, 0), False, self.trajectory, 2)
        
        # 绘制汽车（带方向）
        car_rect = pygame.Rect(self.car_x - self.car_width//2, 
                              self.car_y - self.car_height//2,
                              self.car_width, self.car_height)
        
        # 创建旋转的汽车表面
        car_surface = pygame.Surface((self.car_width, self.car_height), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, self.car_width, self.car_height))
        
        # 绘制方向指示器
        pygame.draw.rect(car_surface, (0, 0, 0), 
                        (self.car_width - 5, self.car_height//2 - 2, 5, 4))
        
        # 旋转汽车
        rotated_car = pygame.transform.rotate(car_surface, -self.car_angle)
        rotated_rect = rotated_car.get_rect(center=car_rect.center)
        
        self.screen.blit(rotated_car, rotated_rect.topleft)
        
        # 显示信息
        font = pygame.font.Font(None, 36)
        info_text = f"Position: ({int(self.car_x)}, {int(self.car_y)}) Angle: {int(self.car_angle)}"
        text_surface = font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """关闭环境"""
        pygame.quit()