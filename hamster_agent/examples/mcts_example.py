#!/usr/bin/env python3
"""
MCTS Agent 使用示例

演示如何使用新的标准MCTS算法进行问题求解
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from hamster_agent.app.agent.hamster_mcts_agent import MCTSAgent, MCTSConfig
    from hamster_agent.app.logger import logger
except ImportError:
    # 尝试相对导入
    from app.agent.hamster_mcts_agent import MCTSAgent, MCTSConfig
    from app.logger import logger


async def example_simple_search():
    """简单的MCTS搜索示例"""
    
    # 配置MCTS参数
    mcts_config = MCTSConfig(
        max_depth=8,              # 最大搜索深度
        iterations=50,            # MCTS迭代次数
        n_generate_samples=3,     # 每次扩展生成的样本数
        exploration_coef=1.414,   # UCB1探索系数
        negative_reward=-1.0,     # 负奖励
        positive_reward=1.0,      # 正奖励
        simulation_depth=3        # 模拟深度
    )
    
    # 创建MCTS代理
    agent = MCTSAgent(mcts_config=mcts_config)
    
    # 执行搜索
    query = "帮我分析一下当前目录下的Python文件，并生成一个总结报告"
    
    logger.info("🚀 开始MCTS搜索...")
    result = await agent.run_mcts_search(
        query=query,
        save_path="mcts_search_result.json"
    )
    
    print(result)
    
    # 显示所有探索的路径
    print(f"\n📋 探索的路径总数: {len(agent.all_paths)}")
    for i, path in enumerate(agent.all_paths[:5]):  # 只显示前5条路径
        print(f"\n路径 {i+1}:")
        for j, node in enumerate(path):
            if node.action:
                print(f"  步骤 {j}: {node.action}")
                print(f"    访问次数: {node.visits}, 价值: {node.value:.3f}")


async def example_load_and_analyze():
    """加载已保存的MCTS结果并分析"""
    
    result_file = "mcts_search_result.json"
    if not Path(result_file).exists():
        print(f"❌ 文件 {result_file} 不存在，请先运行搜索示例")
        return
    
    # 加载结果
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("📊 MCTS搜索结果分析:")
    print(f"- 配置: {data['mcts_config']}")
    print(f"- 统计: {data['statistics']}")
    print(f"- 路径数量: {data['all_paths_count']}")
    print(f"- 最佳路径动作: {data['best_path_actions']}")
    
    # 分析树结构
    def analyze_tree(node_data, depth=0):
        indent = "  " * depth
        action = node_data.get('action', 'root')
        visits = node_data.get('visits', 0)
        value = node_data.get('value', 0)
        avg_value = value / visits if visits > 0 else 0
        
        print(f"{indent}📍 {action} (访问:{visits}, 平均值:{avg_value:.3f})")
        
        # 只显示访问过的子节点
        children = [child for child in node_data.get('children', []) if child.get('visits', 0) > 0]
        children.sort(key=lambda x: x.get('visits', 0), reverse=True)
        
        for child in children[:3]:  # 只显示前3个最佳子节点
            analyze_tree(child, depth + 1)
    
    print("\n🌳 搜索树结构 (仅显示访问过的节点):")
    analyze_tree(data['tree'])


async def example_custom_tools():
    """使用自定义工具的MCTS搜索示例"""
    
    # 创建带基础工具的MCTS代理
    mcts_config = MCTSConfig(
        max_depth=6,
        iterations=30,
        n_generate_samples=2,
        exploration_coef=1.0
    )
    
    agent = MCTSAgent(mcts_config=mcts_config)
    
    # 执行搜索
    query = "帮我创建一个简单的文本文件并写入内容"
    
    result = await agent.run_mcts_search(
        query=query,
        save_path="mcts_custom_tools_result.json"
    )
    
    print(result)


async def example_compare_paths():
    """比较不同路径的效果"""
    
    agent = MCTSAgent()
    
    # 加载已有的搜索结果
    if agent.load_from_file("mcts_search_result.json"):
        print("✅ 成功加载MCTS搜索结果")
        
        # 获取所有路径
        agent._collect_all_paths(agent.root, [])
        
        print(f"\n📈 路径效果分析 (共 {len(agent.all_paths)} 条路径):")
        
        # 按最终价值排序
        path_scores = []
        for i, path in enumerate(agent.all_paths):
            if path:
                final_node = path[-1]
                avg_score = final_node.value / final_node.visits if final_node.visits > 0 else 0
                path_scores.append((i, avg_score, len(path), final_node.visits))
        
        path_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🏆 前10个最佳路径:")
        for rank, (path_idx, score, length, visits) in enumerate(path_scores[:10]):
            path = agent.all_paths[path_idx]
            print(f"  {rank+1}. 路径 {path_idx}: 分数={score:.3f}, 长度={length}, 访问={visits}")
            
            # 显示路径中的关键动作
            actions = [node.action for node in path[1:] if node.action]
            if actions:
                print(f"     动作: {' → '.join(actions[:3])}{'...' if len(actions) > 3 else ''}")
        
        print("\n📉 最差的5条路径:")
        for rank, (path_idx, score, length, visits) in enumerate(path_scores[-5:]):
            path = agent.all_paths[path_idx]
            print(f"  {rank+1}. 路径 {path_idx}: 分数={score:.3f}, 长度={length}, 访问={visits}")


async def main():
    """主函数，演示所有示例"""
    
    print("🎯 MCTS Agent 示例程序")
    print("=" * 50)
    
    try:
        # 示例1: 简单搜索
        print("\n1️⃣ 执行简单MCTS搜索...")
        await example_simple_search()
        
        # 示例2: 分析结果
        print("\n2️⃣ 分析搜索结果...")
        await example_load_and_analyze()
        
        # 示例3: 比较路径
        print("\n3️⃣ 比较不同路径效果...")
        await example_compare_paths()
        
        print("\n✅ 所有示例执行完成!")
        
    except Exception as e:
        logger.error(f"❌ 示例执行出错: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
