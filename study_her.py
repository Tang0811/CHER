import numpy as np
import random
import baselines.cher.config_curriculum as config_cur
import math
from sklearn.neighbors import NearestNeighbors
from gym.envs.robotics import rotations
def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    # 這邊應該是proximity比例的部份
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):

        # 一批50個，批次的內容：key: array(buffer_size x T x dim_key)
        #                           2      x50 x    4
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = config_cur.learning_candidates

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)  # 隨機取proximity比例的數量
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]  # 下一次已完成目標，episode裡面的某次
        transitions['g'][her_indexes] = future_ag

        if batch_size_in_transitions != config_cur.learning_selected:  # 保險
            batch_size_in_transitions = config_cur.learning_selected
        transitions = curriculum(transitions, batch_size_in_transitions, batch_size)  # 根據當下的學習階段去調整
        batch_size = batch_size_in_transitions
        '''call curriculum function'''
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # 用transitions取得reward
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)  # 使用config中的定義計算reward
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }
        assert (transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    def curriculum(transitions, batch_size_in_transitions, batch_size):
        sel_list = lazier_and_goals_sample_kg(transitions['g'], transitions['ag'], transitions['o'],
                                              batch_size_in_transitions)
        transitions = {key: transitions[key][sel_list].copy()
                       for key in transitions.keys()}
        config_cur.learning_step += 1  # 用來紀錄現在到訓練的哪裡了
        return transitions

    def lazier_and_goals_sample_kg(goals, ac_goals, obs, batch_size_in_transitions):  # Algorithm 1 -- Stochastic Greedy
        # ⬇ 看起來沒有用的轉置目標判斷 ⬇
        # if config_cur.goal_type =="ROTATION":
        #     goals,ac_goals=goals[...,3:],ac_goals[...,3:]

        # 128組三維資料下去做相關性的分群
        # NearsetNeighbor使用：找附近1個neighbor
        # 算法使用：kd tree
        # 計算距離方式：euclidean
        num_neighbor = 1
        kgraph = NearestNeighbors(n_neighbors=num_neighbor, algorithm='kd_tree', metric='euclidean').fit(
            goals).kneighbors_graph(mode=
                                    'distance').tocoo(copy=False)
        row = kgraph.row  # row 数组包含"起始"节点的索引
        col = kgraph.col  # col 数组包含"目标"节点的索引
        sim = np.exp(-np.divide(np.power(kgraph.data, 2), np.mean(kgraph.data) ** 2))
        # 因為num_neighbor數=1，所以sim就是最近的1個neighbor距離轉換的相似分數
        delta = np.mean(kgraph.data)

        sel_idx_set = []  # 等待選擇的資料堆
        idx_set = [i for i in range(len(goals))]  # idx
        balance = config_cur.fixed_lambda  # 調配的比例，從config取得，-1
        if int(balance) == -1:
            balance = math.pow(1 + config_cur.learning_rate,
                               config_cur.learning_step) * config_cur.lambda_starter  # ((1+0.0001)^step)*1
            v_set = [i for i in range(len(goals))]
            max_set = []
            for i in range(0, batch_size_in_transitions):  # 0~64
                sub_size = 3
                sub_set = random.sample(idx_set, sub_size)  # 0~128隨機選3
                sel_idx = -1
                max_marginal = float("-inf")
                for j in range(sub_size):  # 執行三次
                    k_idx = sub_set[j]  # 第1/2/3個
                    marginal_v, new_a_set = fa(k_idx, max_set, v_set, sim, row, col)
                    euc = np.linalg.norm(goals[sub_set[j]] - ac_goals[sub_set[j]])  # 兩個點的歐氏距離
                    marginal_v = marginal_v - balance * euc
                    if marginal_v > max_marginal:  # 比較三次取出最大的
                        sel_idx = k_idx
                        max_marginal = marginal_v
                        max_set = new_a_set

                idx_set.remove(sel_idx)
                sel_idx_set.append(sel_idx)
            return np.array(sel_idx_set)

    def fa(k, a_set, v_set, sim, row, col):
        if len(a_set) == 0:  # 一開始輸入是空集合，也就是執行第一次sub_size的時候
            init_a_set = []
            marginal_v = 0
            for i in v_set:  # 看過所有128比資料
                max_ki = 0
                if k == col[i]:  # 如果隨擠出來的3個等於其他人的NearsNeighbor的話
                    max_ki = sim[i]  # 計算拿出那一個人的sim分數
                init_a_set.append(max_ki)  # 0,0,0,...,0,sim[i],sim[i],...
                marginal_v += max_ki  # 總和
            return marginal_v, init_a_set
        # 在執行第2/3次
        new_a_set = []
        marginal_v = 0
        for i in v_set:  # 看過所有128比資料
            sim_ik = 0
            if k == col[i]:  # 如果隨擠出來的3個等於其他人的NearsNeighbor的話
                sim_ik = sim[i]  # 計算拿出那一個人的sim分數

            # a_set應該是前面次中的最大值，如果比較大
            if sim_ik > a_set[i]:
                max_ki = sim_ik  # 如果比較大，替換
                new_a_set.append(max_ki)
                marginal_v += max_ki - a_set[i]  # 加上：最大的減掉次大的
            # 如果當下的沒有比較大
            else:
                new_a_set.append(a_set[i])
        return marginal_v, new_a_set


    return _sample_her_transitions


'''
    def plot_kgraph(kgraph,row,col):
        # 使用方法
        # plot_kgraph(kgraph, row, col)

        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np

        # 创建一个空的有向图
        graph = nx.DiGraph()

        # 添加边到图中
        for i in range(len(row)):
            weight = kgraph.data[i]
            if weight == 0:
                edge_color = 'gray'  # 将边的颜色设置为灰色
            else:
                edge_color = 'black'  # 默认边的颜色为黑色
            graph.add_edge(row[i], col[i], weight=weight, color=edge_color)

        # 设置图形的大小
        plt.figure(figsize=(10, 6))

        # 绘制图形
        pos = nx.spring_layout(graph)  # 选择节点布局算法

        # 绘制节点
        nx.draw_networkx_nodes(graph, pos)

        # 绘制边，并设置边的颜色
        edge_colors = [graph[u][v]['color'] for u, v in graph.edges()]
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors)

        # 绘制节点的标签
        node_labels = {node: str(node) for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=node_labels)

        # # 绘制边的权重标签
        # edge_labels = nx.get_edge_attributes(graph, 'weight')
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        # 调整图形布局，以便标签不重叠
        plt.tight_layout()

        # 显示图形
        plt.axis('off')

        plt.savefig('/home/jeff/CHER/baselines/cher/experiment/filename.png', dpi=300)

    def plot_kgraph_pca(kgraph, goals):
        import networkx as nx
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # 使用 PCA 降維
        pca = PCA(n_components=2)
        goals_2d = pca.fit_transform(goals)

        # 创建一个空的有向图
        graph = nx.DiGraph()

        # 添加边到图中
        for i, j, v in zip(kgraph.row, kgraph.col, kgraph.data):
            if v == 0:
                edge_color = 'gray'  # 将边的颜色设置为灰色
            else:
                edge_color = 'black'  # 默认边的颜色为黑色
            graph.add_edge(i, j, weight=v, color=edge_color)

        # 设置图形的大小
        plt.figure(figsize=(10, 6))

        # 选择节点布局算法，使用 PCA 降维后的坐标
        pos = {i: goals_2d[i] for i in range(len(goals_2d))}

        # 绘制节点
        nx.draw_networkx_nodes(graph, pos)

        # 绘制边，并设置边的颜色
        edge_colors = [graph[u][v]['color'] for u, v in graph.edges()]
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors)

        # 绘制节点的标签
        node_labels = {node: str(node) for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=node_labels)

        # 调整图形布局，以便标签不重叠
        plt.tight_layout()

        # 显示图形
        plt.axis('off')

        # 将图像保存到指定路径
        plt.savefig('/home/jeff/CHER/baselines/cher/experiment/filename.png', dpi=300)
'''
