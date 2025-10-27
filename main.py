import os
import sys
import argparse
from car_env import CarEnvironment
from models import BehaviorCloningModel, CNNBehaviorCloningModel, EnsembleModel
from data_collector import DataCollector
from trainer import BCTrainer
from tester import BCTester

def main():
    parser = argparse.ArgumentParser(description='模仿学习自动驾驶小车')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['collect', 'train', 'test', 'demo'],
                       help='运行模式: collect(收集数据), train(训练), test(测试), demo(演示)')
    parser.add_argument('--model-type', type=str, default='state',
                       choices=['state', 'pixel', 'ensemble'],
                       help='模型类型: state(状态向量), pixel(图像), ensemble(集成)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='episode数量')
    parser.add_argument('--data-file', type=str,
                       help='数据文件路径')
    parser.add_argument('--model-file', type=str,
                       help='模型文件路径')
    
    args = parser.parse_args()
    
    # 创建环境
    env = CarEnvironment(randomize=True, seed=42)
    
    try:
        if args.mode == 'collect':
            # 数据收集模式
            collector = DataCollector(env)
            collector.collect_multiple_episodes(args.episodes)
            
            # 保存数据
            data_file = collector.save_dataset()
            print(f"数据已保存: {data_file}")
            
            # 显示统计信息
            stats = collector.get_dataset_stats()
            print("\n数据集统计:")
            print(stats)
        
        elif args.mode == 'train':
            # 训练模式
            if not args.data_file:
                print("请提供数据文件路径 --data-file")
                return
            
            # 加载数据
            collector = DataCollector(env)
            collector.load_dataset(args.data_file)
            
            # 创建模型
            if args.model_type == 'state':
                model = BehaviorCloningModel()
            elif args.model_type == 'pixel':
                model = CNNBehaviorCloningModel()
            elif args.model_type == 'ensemble':
                model = EnsembleModel()
            
            # 训练
            trainer = BCTrainer(model, args.model_type)
            train_loader, val_loader = trainer.prepare_data(collector.dataset)
            trainer.train(train_loader, val_loader, epochs=100)
            
            # 保存模型和训练历史
            model_file = trainer.save_model()
            trainer.plot_training_history()
            
            print(f"训练完成! 模型已保存: {model_file}")
        
        elif args.mode == 'test':
            # 测试模式
            if not args.model_file:
                print("请提供模型文件路径 --model-file")
                return
            
            # 创建模型
            if args.model_type == 'state':
                model = BehaviorCloningModel()
            elif args.model_type == 'pixel':
                model = CNNBehaviorCloningModel()
            elif args.model_type == 'ensemble':
                model = EnsembleModel()
            
            # 加载模型
            trainer = BCTrainer(model, args.model_type)
            trainer.load_model(args.model_file)
            
            # 测试
            tester = BCTester(env, model, args.model_type)
            results = tester.test_multiple_episodes(args.episodes)
        
        elif args.mode == 'demo':
            # 演示模式 - 同时展示专家和AI驾驶
            print("演示模式: 观察AI自动驾驶表现")
            
            if not args.model_file:
                print("请提供模型文件路径 --model-file")
                return
            
            # 创建并加载模型
            if args.model_type == 'state':
                model = BehaviorCloningModel()
            elif args.model_type == 'pixel':
                model = CNNBehaviorCloningModel()
            elif args.model_type == 'ensemble':
                model = EnsembleModel()
            
            trainer = BCTrainer(model, args.model_type)
            trainer.load_model(args.model_file)
            
            tester = BCTester(env, model, args.model_type)
            tester.test_episode(max_steps=500, render=True)
    
    finally:
        env.close()

if __name__ == "__main__":
    main()