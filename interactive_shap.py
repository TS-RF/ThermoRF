#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式SHAP分析工具

功能：
1. 用户选择要分析的故障类型（单个或多个）
2. 用户选择要分析的样本
3. 生成所有类型的SHAP图表：
   - Waterfall图（单样本解释）
   - Beeswarm图（全局特征重要性）
   - Summary Plot（特征分布）
   - Composite图（组合视图）
   - Dependence图（单变量）
   - Dependence图（双变量 ⭐）
   - Interaction图（特征交互）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data, prepare_data, get_class_names
from src.models import RandomForestModel
from src.shap_analysis import SHAPAnalyzer
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# 故障类型名称
FAULT_TYPES = {
    0: "Normal (正常)",
    1: "Head-crack (气缸盖裂纹)",
    2: "Linner-wear (缸套磨损)",
    3: "Piston-ablation (活塞烧蚀)",
    4: "Ring-wear (活塞环磨损)",
    5: "Ring-adhesion (活塞环粘连)"
}

def print_separator(char='=', length=70):
    """打印分隔线"""
    print(char * length)

def display_menu():
    """显示主菜单"""
    print_separator()
    print("🔬 交互式SHAP分析工具")
    print_separator()
    print("\n可用的故障类型：")
    for idx, name in FAULT_TYPES.items():
        print(f"  [{idx}] {name}")
    print()

def get_user_choice(prompt, valid_choices):
    """获取用户选择"""
    while True:
        try:
            choice = input(prompt)
            if choice.lower() == 'q':
                print("退出程序")
                sys.exit(0)
            
            # 处理多个选择（逗号分隔）
            if ',' in choice:
                choices = [int(x.strip()) for x in choice.split(',')]
                if all(c in valid_choices for c in choices):
                    return choices
                else:
                    print(f"❌ 无效选择。请输入 {valid_choices} 中的值")
            else:
                choice_int = int(choice)
                if choice_int in valid_choices:
                    return [choice_int]
                else:
                    print(f"❌ 无效选择。请输入 {valid_choices} 中的值")
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n退出程序")
            sys.exit(0)

def select_plot_types():
    """选择要生成的图表类型"""
    print("\n选择要生成的SHAP图表类型：")
    print("  [1] Waterfall图 - 单样本预测解释")
    print("  [2] Beeswarm图 - 全局特征重要性分布")
    print("  [3] Composite图 - 组合视图（重要性+分布）")
    print("  [4] Dependence图（双变量）⭐ - 特征交互效应")
    print("  [5] Interaction图 - 特征交互强度排名")
    print("  [0] 生成所有图表")
    
    choice = input("\n请输入选择（多个用逗号分隔，如'1,2,4'）[默认0-所有]: ").strip()
    
    if not choice or choice == '0':
        return [1, 2, 3, 4, 5]
    
    try:
        choices = [int(x.strip()) for x in choice.split(',')]
        return [c for c in choices if 1 <= c <= 5]
    except:
        return [1, 2, 3, 4, 5]

def main():
    display_menu()
    
    # 1. 选择故障类型
    print("━" * 70)
    print("步骤1：选择要分析的故障类型")
    print("━" * 70)
    fault_indices = get_user_choice(
        "请输入故障类型编号（多个用逗号分隔，如'0,2,4'）[默认0]: ",
        list(FAULT_TYPES.keys())
    )
    if not fault_indices:
        fault_indices = [0]
    
    print(f"\n✓ 已选择: {[FAULT_TYPES[i] for i in fault_indices]}")
    
    # 2. 选择图表类型
    print("\n━" * 70)
    print("步骤2：选择SHAP图表类型")
    print("━" * 70)
    plot_types = select_plot_types()
    
    plot_names = {
        1: "Waterfall", 2: "Beeswarm", 3: "Composite",
        4: "Dependence(单变量)", 5: "Dependence(双变量)", 6: "Interaction"
    }
    print(f"\n✓ 已选择: {[plot_names[p] for p in plot_types]}")
    
    # 3. 加载和准备数据
    print("\n" + "=" * 70)
    print("数据加载与模型训练")
    print("=" * 70)
    
    use_feature_selection = input("\n是否使用特征选择？[y/N]: ").strip().lower() == 'y'
    random_state = 20
    
    print("\n📊 正在加载数据...")
    X, y, label_encoder, _ = load_data(DATA_DIR, use_feature_selection=use_feature_selection)
    print(f"   ✓ 数据形状: {X.shape}")
    print(f"   ✓ 特征数: {X.shape[1]}")
    if hasattr(X, 'columns'):
        print(f"   ✓ 特征名: {list(X.columns)}")
    
    print("\n📊 正在准备数据...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        X, y, test_size=216, random_state=random_state, normalize=True
    )
    print(f"   ✓ 训练集: {X_train.shape}")
    print(f"   ✓ 测试集: {X_test.shape}")
    
    # 4. 训练模型
    print("\n🤖 正在训练随机森林模型...")
    rf = RandomForestModel(n_estimators=20, random_state=random_state)
    rf.train(X_train, y_train)
    
    class_names = get_class_names(label_encoder)
    accuracy = rf.evaluate(X_test, y_test, class_names)['accuracy']
    print(f"   ✓ 模型准确率: {accuracy*100:.2f}%")
    
    # 5. 初始化SHAP分析器
    print("\n⚙️  正在初始化SHAP分析器...")
    feature_names = list(X.columns) if hasattr(X, 'columns') else None
    shap_analyzer = SHAPAnalyzer(
        model=rf.model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names
    )
    print("   ✓ SHAP值计算完成")
    
    # 获取每个类别的测试样本数量
    print("\n📊 测试集中各类别的样本数量：")
    for class_idx in fault_indices:
        count = np.sum(y_test == class_idx)
        print(f"   {FAULT_TYPES[class_idx]}: {count}个样本")
    
    # 6. 生成SHAP图表
    print("\n" + "=" * 70)
    print("生成SHAP可视化图表")
    print("=" * 70)
    
    for class_idx in fault_indices:
        print(f"\n{'━' * 70}")
        print(f"正在分析: {FAULT_TYPES[class_idx]}")
        print(f"{'━' * 70}")
        
        # Waterfall图
        if 1 in plot_types:
            # 找出该类别的测试样本
            class_samples = np.where(y_test == class_idx)[0]
            if len(class_samples) > 0:
                print(f"\n该类别有 {len(class_samples)} 个测试样本")
                sample_choice = input(f"选择样本索引 [0-{len(class_samples)-1}，默认0]: ").strip()
                sample_idx = int(sample_choice) if sample_choice.isdigit() else 0
                sample_idx = min(sample_idx, len(class_samples)-1)
                
                actual_sample_idx = class_samples[sample_idx]
                print(f"\n📊 生成Waterfall图（样本 #{actual_sample_idx}）...")
                shap_analyzer.plot_waterfall(
                    class_idx=class_idx,
                    sample_idx=actual_sample_idx,
                    output_dir=OUTPUT_DIR
                )
                print(f"   ✓ SHAP_waterfall_F{class_idx}_sample{actual_sample_idx}.png")
            else:
                print(f"   ⚠️  测试集中没有类别{class_idx}的样本，跳过Waterfall图")
        
        # Beeswarm图
        if 2 in plot_types:
            print(f"\n📊 生成Beeswarm图...")
            shap_analyzer.plot_beeswarm(
                class_idx=class_idx,
                output_dir=OUTPUT_DIR
            )
            print(f"   ✓ SHAP_beeswarm_F{class_idx}.png")
        
        # Composite图
        if 3 in plot_types:
            print(f"\n📊 生成Composite图...")
            shap_analyzer.plot_composite(
                class_idx=class_idx,
                output_dir=OUTPUT_DIR
            )
            print(f"   ✓ SHAP_composite_F{class_idx}.png")
        
        # 双变量Dependence图 ⭐
        if 4 in plot_types:
            print(f"\n⭐ 生成双变量Dependence图（特征交互效应）...")
            
            # 显示所有特征及其编号
            all_features = list(X.columns) if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
            print("\n可用特征编号：")
            for i, feat_name in enumerate(all_features):
                print(f"  P{i+1:02d}: {feat_name}")
            
            # 获取top特征用于推荐
            feature_importance = np.abs(shap_analyzer.shap_values_numpy[..., class_idx]).mean(0)
            top_features_idx = np.argsort(feature_importance)[-4:][::-1]
            top_features_nums = [f"P{i+1:02d}" for i in top_features_idx]
            top_features_names = [all_features[i] for i in top_features_idx]
            
            print(f"\n💡 推荐（Top-4重要特征）：")
            for num, name in zip(top_features_nums, top_features_names):
                print(f"  {num}: {name}")
            
            # 手动指定特征对（使用编号）
            print("\n请输入特征对（使用编号P01-P14）：")
            print("示例: P01-P02,P03-P04  或  P01-P05")
            pairs_input = input("特征对（多个用逗号分隔）: ").strip()
            
            if pairs_input:
                pairs = [pair.strip().split('-') for pair in pairs_input.split(',')]
                print("\n开始生成...")
                for feat_x_num, feat_y_num in pairs:
                    feat_x_num = feat_x_num.strip().upper()
                    feat_y_num = feat_y_num.strip().upper()
                    
                    try:
                        # 解析编号（P01 -> 0, P02 -> 1, ...）
                        if feat_x_num.startswith('P') and feat_y_num.startswith('P'):
                            x_idx = int(feat_x_num[1:]) - 1
                            y_idx = int(feat_y_num[1:]) - 1
                            
                            if 0 <= x_idx < len(all_features) and 0 <= y_idx < len(all_features):
                                feat_x = all_features[x_idx]
                                feat_y = all_features[y_idx]
                                
                                shap_analyzer.plot_dependence(
                                    feature_x=feat_x,
                                    feature_y=feat_y,
                                    class_idx=class_idx,
                                    output_dir=OUTPUT_DIR
                                )
                                print(f"   ✓ {feat_x_num}({feat_x}) vs {feat_y_num}({feat_y})")
                            else:
                                print(f"   ✗ {feat_x_num}-{feat_y_num} - 编号超出范围（P01-P{len(all_features):02d}）")
                        else:
                            print(f"   ✗ {feat_x_num}-{feat_y_num} - 格式错误（应为P01-P14格式）")
                    except ValueError:
                        print(f"   ✗ {feat_x_num}-{feat_y_num} - 编号格式错误")
                    except Exception as e:
                        print(f"   ✗ {feat_x_num}-{feat_y_num} - 错误: {str(e)}")
            else:
                print("   ⚠️  未输入特征对，跳过双变量图生成")
    
    # Interaction图（只生成一次）
    if 5 in plot_types:
        representative_class = fault_indices[len(fault_indices)//2]
        print(f"\n{'━' * 70}")
        print(f"📊 生成Interaction图（代表类别: {FAULT_TYPES[representative_class]}）...")
        print(f"{'━' * 70}")
        shap_analyzer.plot_interaction(
            class_idx=representative_class,
            output_dir=OUTPUT_DIR
        )
        print(f"   ✓ SHAP_interaction_F{representative_class}.png")
    
    # 7. 总结
    print("\n" + "=" * 70)
    print("✅ SHAP分析完成！")
    print("=" * 70)
    print(f"\n📁 所有图表已保存到: {OUTPUT_DIR}/")
    
    # 显示生成的文件
    import glob
    shap_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'SHAP_*.png')))
    print(f"\n共生成 {len(shap_files)} 个SHAP图表：")
    for f in shap_files[-20:]:  # 显示最近20个
        print(f"  ✓ {os.path.basename(f)}")
    
    if len(shap_files) > 20:
        print(f"  ... 还有 {len(shap_files)-20} 个文件")
    
    print("\n💡 提示：")
    print("  - 双变量图文件名包含 'bivariate' 关键字")
    print("  - 颜色代表第二个特征的值，观察颜色规律判断交互效应")
    print("  - 更多信息请查看 SHAP_BIVARIATE_GUIDE.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
