# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:34:49 2021

@author: ZD

CART-Tree
"""
from collections import Counter
import numpy as np
import pandas as pd

# 定义树节点
class Node:
    _name = None #当前节点名称，即特征名称
    _type = None #节点类型，root:根节点；node:普通节点；leaf:叶节点
    _split_value = None #当前节点向下切分的值
    _from_father = None #当前节点的父节点名称
    _to_left_son = None #当前节点向下划分的左节点名称
    _to_right_son = None #当前节点向下划分的右节点名称
    _output = None #若为叶节点，则为具体的结果输出；否则为None

def _extract(data_list):
    statistic = Counter(data_list)
    max_num = 0
    item = None
    for key in list(statistic.keys()):
        if statistic[key] > max_num:
            max_num = statistic[key]
            item = key
    return item

def _measure(data):
    """
    data:dataframe, column_name = ['y_true','y_pred','tag']
    """
    class_list = data['y_true'].drop_duplicates().tolist()
    F_list = []
    for c in class_list:
        temp = data[data.y_true == c]
        R = len(temp[temp.tag == 1])/len(temp)
        
        _temp = data[data.y_pred == c]
        if len(_temp) <= 0:
            P = 0
        else:
            P = len(_temp[_temp.tag == 1])/len(_temp)
        f_1 = (2*R*P)/(P + R)
        F_list.append(f_1)
    return np.mean(F_list)


# Gini-based CART Tree for classifier
class DecisionTreeClassifier:
    
    def __init__(self, dataInput, dataValid, con_feature = None, gini_threshold = 1.0e-6, sample_threshold = 10, max_depth = 100):
        """
        param dataInput: dataframe,training data（最后一列为类别）,训练数据集
        param dataValid: 验证数据集，其他同dataInput
        param con_feature: list，（指定连续型特征所在列的名称）
        param gini_threshold: float，基尼系数的阈值（基尼指数越小，不确定性越小（不纯度越小））
        param sample_threshold: positive interger，最小样本量的阈值
        """
        self.features = dataInput.values[:,:-1]#array,存储样本的特征值
        self.labels = dataInput.values[:,-1]#array,存储样本的类别
        self.valid_data = dataValid.to_dict('records')#转成dict格式
        self.gini_threh = gini_threshold
        self.sample_threh = sample_threshold
        self.sample_num = len(dataInput)#样本量
        self.tree_depth = max_depth # 树的最大深度阈值
        self.column_name = list(dataInput.columns)
        self.columns = self.column_name[:-1] # 存储各特征列的名称
        
        self.feature_tag = [] # 存储每个特征的属性：离散型 0，连续型 1
        if con_feature == None:
            self.feature_tag = [0]*len(self.columns)
        else:
            for x in self.columns:
                if x in con_feature:
                    self.feature_tag.append(1)
                else:
                    self.feature_tag.append(0)
    
    def Gini_calc(self,data):
        #data:list,不同类别的数据集
        #return：gini index
        total_num = len(data)
        count = Counter(data)
        alpha = 0.0
        for item in count.keys():
            alpha = alpha + (count[item]/total_num)**2
        gini = 1-alpha
        return gini
    
    # calculate the gini index of disgrete feature
    def feature_gini_discrete(self,feature_data,y_data):
        # x_data: array;
        # y_data: list;
        # feature_select: positive interger
        #feature_data = list(x_data[:,feature_select])
        # feature_data: list, discrete value
        feature_value = list(set(feature_data))#特征变量中所含有的不同特征值
        total_num = len(feature_data)
        
        if total_num <= self.sample_threh:
            return None, None, None
        
        min_gini = 1.0#存储最小基尼指数
        value_gini = {}#储存每个特征值的基尼指数
        best_value = feature_value[0] # 存储基尼系数最小的特征值
        
        for value in feature_value:
            middle = {}
            left_index_list = [index for index, x in enumerate(feature_data) if x == value]
            if len(left_index_list) <= self.sample_threh:
                continue
            left_tag_list = [y_data[index] for index in left_index_list]
            right_tag_list = [y_data[i] for i in range(len(y_data)) if i not in left_tag_list]
            
            if len(right_tag_list) <= self.sample_threh:
                continue
            
            sp1_gini = self.Gini_calc(left_tag_list)
            sp2_gini = self.Gini_calc(right_tag_list)
            gini = (len(left_tag_list)/total_num) * sp1_gini + (len(right_tag_list)/total_num) * sp2_gini
            middle['index'] = left_index_list
            middle['gini'] = gini
            middle['value'] = value
            
            if gini < min_gini:
                min_gini = gini
                best_value = value
                value_gini[value] = middle
        if min_gini >= 1.0:
            return None, None, None
        return min_gini, best_value, value_gini
    
    #calculate the gini index of continues feature
    def feature_gini_continue(self,feature_data, y_data):
        
        # feature_data: list;
        # y_data: list;
        total_num = len(feature_data)
        
        if total_num <= self.sample_threh:
            return None, None, None
        
        value_gini = {}#storing the gini index based on the different feature values
        
        point_temp = sorted(list(set(feature_data.copy())))
        
        # obtaining the possible split value
        splited_point = []
        for i in range(len(point_temp) - 1):
            point = (point_temp[i] + point_temp[i+1])/2
            splited_point.append(point)
        
        # determin the optimal split value based on the gini index
        best_value = splited_point[0]
        min_gini = 1.0
        for split in splited_point:
            sp1_tag_data = []
            sp2_tag_data = []
            middle = {}
            left_index_list = []
            for index, x in enumerate(feature_data):
                if x<= split:
                    sp1_tag_data.append(y_data[index])
                    left_index_list.append(index)
                else:
                    sp2_tag_data.append(y_data[index])
            if (len(sp1_tag_data) <= self.sample_threh) or (len(sp2_tag_data) <= self.sample_threh):
                continue
            sp1_gini = self.Gini_calc(sp1_tag_data)
            sp2_gini = self.Gini_calc(sp2_tag_data)
            gini = (len(sp1_tag_data)/total_num)*sp1_gini + (len(sp2_tag_data)/total_num)*sp2_gini
            
            middle['index'] = left_index_list
            middle['gini'] = gini
            middle['value'] = split
            
            if gini < min_gini:
                best_value = split
                min_gini = gini
                value_gini = middle
        return min_gini, best_value, value_gini
    
    
    def _choose_feature(self, feature_data_all, y_data_all, feature_tag):
        """
        feature_data_all: array
        y_data: list
        feature_tag: list
        """
        split_tag = True
        feature_num = feature_data_all.shape[1]
        opt_gini = 1.0
        opt_feature_index = 0
        choose_feature_value = {}
        for i in range(feature_num):
            feature_data = list(feature_data_all[:,i])
            if feature_tag[i] == 0:
                best_value_gini, best_value, value_gini = self.feature_gini_discrete(feature_data,y_data_all)
            else:
                best_value_gini, best_value, value_gini = self.feature_gini_continue(feature_data,y_data_all)
            if best_value_gini == None:
                continue
            else:
                opt_gini = best_value_gini
                opt_feature_index = i
                choose_feature_value = value_gini[best_value]
        if (opt_gini >= 1.0):
            return split_tag, None,None,None
        
        if opt_gini <= self.gini_threh:
            split_tag = False
        
        feature_value = choose_feature_value['value']
        left_index_list = choose_feature_value['index']
        right_index_list = [x for x in len(feature_data_all) if x not in left_index_list]
        
        left_feature = [list(feature_data_all[j,:]) for j in left_index_list]
        left_y = [y_data_all[m] for m in left_index_list]
        
        right_feature = [list(feature_data_all[j,:]) for j in right_index_list]
        right_y = [y_data_all[m] for m in right_index_list]
        return opt_gini, split_tag, [opt_feature_index, feature_value], [np.array(left_feature), left_y], [np.array(right_feature), right_y]
    
    def _creat_Initaltree(self):
        init_tree = {}
        init_feature = self.features.copy()
        init_y_data = self.labels.copy
        feature_tag = self.feature_tag.copy()
        feature_gini, root_feature, left_data, right_data = self._choose_feature(init_feature, list(init_y_data), feature_tag)
        
        root0 = {}
        root0['feature_index'] = root_feature[0]
        root0['feature_name'] = self.columns[root_feature[0]]
        root0['feature_gini'] = feature_gini
        root0['split_value'] = root_feature[1]
        root0['left_data'] = left_data
        root0['right_data'] = right_data
        root0['split'] = True
        
        init_tree['root'] = [root0]
        father_level = 'root'
        
        count = 0
        
        while True:
            count = count+1
            level_name = 'level_{}'.format(count)
            father_nodes = init_tree[father_level].copy()
            nodes_list = []
            for f_node in father_nodes:
                
                if not f_node['split']:
                    continue
                
                left_features = f_node['left_data'][0]
                left_y = f_node['left_data'][1]
                
                feature_gini_left, is_split_left, left_feature, left_left_data, left_right_data = self._choose_feature(left_features, left_y, feature_tag)
                if left_feature != None:
                    node = {}
                    node['father'] = f_node['feature_name']
                    node['leaf_dir'] = 'father_left'
                    node['feature_index'] = left_feature[0]
                    node['feature_name'] = self.columns[left_feature[0]]
                    node['feature_gini'] = feature_gini_left
                    node['split_value'] = left_feature[1]
                    node['left_data'] = left_left_data
                    node['right_data'] = left_right_data
                    node['split'] = is_split_left
                    nodes_list.append(node)
                
                right_features = f_node['right_data'][0]
                right_y = f_node['right_data'][1]
                
                feature_gini_right, is_split_right, right_feature, right_left_data, right_right_data = self._choose_feature(right_features, right_y, feature_tag)
                if right_feature != None:
                    node = {}
                    node['father'] = f_node['feature_name']
                    node['leaf_dir'] = 'father_right'
                    node['feature_index'] = right_feature[0]
                    node['feature_name'] = self.columns[right_feature[0]]
                    node['feature_gini'] = feature_gini_right
                    node['split_value'] = right_feature[1]
                    node['left_data'] = right_left_data
                    node['right_data'] = right_right_data
                    node['split'] = is_split_right
                    nodes_list.append(node)
            
            if ((len(nodes_list) <= 0) or (count >= self.tree_depth)):
                break
            
            father_level = level_name
            init_tree[level_name] = nodes_list
        return init_tree
    
    def _leaf_node(self, tree, root_name, next_levels):
        leaf_gini = []
        father_node = [root_name]
        for i_index, level_name in enumerate(next_levels):
            if len(father_node) <= 0:
                break
            nodes = tree[level_name]
            node_temp = []
            for node in nodes:
                if node['father'] in father_node:
                    node_temp.append(node)
            if len(node_temp) <= 0:
                break
            father_node = []
            for tt in node_temp:
                if not tt['split']:
                    tt_data = tt['left_data'][1] + tt['right_data'][1]
                    tt_gini = self.Gini_calc(tt_data)
                    leaf_gini.append(tt_gini)
                else:
                    father_node.append(tt['father'])
        return leaf_gini
    
    def _tree_pruning(self,_init_tree):
        pruned_tree = _init_tree.copy()
        alpha_list = []
        tree_list = []
        while True:
            alpha = 99999
            node_root_index = -1
            root_node_level = '-1'
            node_levels = list(pruned_tree.keys())
            if ((len(node_levels) <= 2) and (node_levels[0] == 'root')):
                break
            # finding minimum alpha
            for node_level in node_levels:
                nodes = pruned_tree[node_level]
                next_levels = node_levels[node_levels.index(node_level)+1:]
                for node_index, node in enumerate(nodes):
                    if node['split']:
                        node_gini = node['feature_gini']
                        father_node = node['feature_name']
                        leaf_gini = self._leaf_node(pruned_tree, father_node, next_levels)
                        alpha_temp = (node_gini - sum(leaf_gini))/(len(leaf_gini) - 1)
                        if alpha_temp < alpha:
                            alpha = alpha_temp
                            node_root_index = node_index
                            root_node_level = node_level
            
            # pruning tree based on alpha
            root_level_index = node_levels.index(root_node_level)
            root_next_node = node_levels[root_level_index+1:]
            
            tree_temp = {}
            root_node = pruned_tree[root_node_level][node_root_index]
            father_node = root_node['feature_name']
            for l_index, level in enumerate(node_levels):
                if l_index < root_level_index:
                    tree_temp[level] = pruned_tree[level]
                else:
                    
                    root_node['split'] = False
                    tree_temp[level] = root_node
                    break
            father_list = [father_node]
            for level in root_next_node:
                level_nodes = pruned_tree[root_next_node]
                father_temp = []
                level_node_temp = []
                for node in level_nodes:
                    if node['father'] in father_list:
                        father_temp.append(node['feature_name'])
                    else:
                        level_node_temp.append(node)
                if len(level_node_temp) <= 0:
                    break
                else:
                    tree_temp[level] = level_node_temp
                father_list = father_temp
            pruned_tree = {}
            pruned_tree = tree_temp
            alpha_list.append(alpha)
            tree_list.append(tree_temp)
        return alpha_list, tree_list
    
    def _tree_decode(self, tree):
        tree = {}
        root_nodes = []
        general_nodes = []
        key_list = list(tree.keys())
        for cur_index, key in enumerate(key_list):
            cur_nodes = tree[key]
            if cur_index < (len(key_list)-1):
                next_nodes = tree[key_list[cur_index + 1]]
            node_decode_list = []
            for node in cur_nodes:
                cur_node = Node
                if cur_index > 0:
                    cur_node._from_father = node['father']
                cur_node._name = node['feature_name']
                cur_node._split_value = node['split_value']
                
                output = None
                if node['split'] == False:
                    cur_node._type = 'leaf'
                    output = _extract(node['left_data'][1] + node['right_data'][1])
                    cur_node._output = output
                else: 
                    cur_node._type = 'node'
                    left_son = None
                    right_son = None
                    for n_node in next_nodes:
                        if (n_node['father'] == cur_node._name) and (n_node['leaf_dir'] == 'father_left'):
                            left_son = n_node['feature_name']
                        elif (n_node['father'] == cur_node._name) and (n_node['leaf_dir'] == 'father_right'):
                            right_son = n_node['feature_name']
                    cur_node._to_left_son = left_son
                    cur_node._to_right_son = right_son
                node_decode_list.append(cur_node)
            if key == 'root':
                root_nodes = node_decode_list
            else:
                general_nodes = general_nodes + node_decode_list
        tree['root'] = root_nodes[0]
        tree['nodes'] = general_nodes
        return tree
    
    def _predict(self, tree, data_dict):
        y_pred = None
        root_node = tree['root']
        gen_nodes_list = tree['nodes']
        
        node_name = root_node._name
        node_split_value = root_node._split_value
        #node_father = root_node._from_father
        node_left_son = root_node._to_left_son
        node_right_son = root_node._to_right_son
        node_output = root_node._output
        
        while True:
            if node_output == None:
                node_value = data_dict[node_name]
                if node_value <= node_split_value:
                    node_son_name = node_left_son
                else:
                    node_son_name = node_right_son
                for gen_node in gen_nodes_list:
                    if (gen_node._name == node_son_name) and (gen_node._from_father == node_name):
                        break
                
                node_name = gen_node._name
                node_split_value = gen_node._split_value
                #node_father = gen_node._from_father
                node_left_son = gen_node._to_left_son
                node_right_son = gen_node._to_right_son
                node_output = gen_node._output
                
            else:
                y_pred = node_output
                break
        
        return y_pred
    
    def _post_pruning(self, init_tree):
        alpha_list, tree_list = self._tree_pruning(init_tree)
        
        # 基于验证集,采用交叉验证确定最优树
        model_F1_list = []
        opt_f1 = 0.0
        opt_tree = None
        for tree in tree_list:
            decoded_tree = self._tree_decode(tree)
            pred_result_list = []
            for record in self.valid_data:
                y_true = record[self.column_name[-1]]
                y_pred = self._predict(decoded_tree, record)
                tag = 1
                if y_true != y_pred:
                    tag = 0
                pred_result_list.append([y_true, y_pred, tag])
            pred_result = pd.DataFrame(pred_result_list, columns = ['y_true', 'y_pred', 'tag'])
            f1 = _measure(pred_result)
            model_F1_list.append(f1)
            if f1 > opt_f1:
                opt_f1 = f1
                opt_tree = tree
        return opt_tree, opt_f1
    
    def train(self):
        init_tree = self._creat_Initaltree()
        classifier_tree, f1 = self._post_pruning(init_tree)
        return classifier_tree
