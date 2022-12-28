import networkx as nx
import pandas as pd
import numpy as np
import math
import itertools
from tqdm import tqdm
from copy import deepcopy


def build_graph(df):
    adj_total = []
    from_list = df['from'].tolist()
    to_list = df['to'].tolist()
    for idx, row in df.iterrows():
        start = row['from']
        to = row['to']
        adj = [[idx, i] for i in range(len(from_list)) if (start < to_list[i] and to > from_list[i])]
        adj_total.extend(adj)
    return np.array(adj_total)
            

def coloring_algorithm(G):
    if len(G) == 0:
        return {}
    colors = {}
    nodes = DSatur(G, colors)
    for u in nodes:
        neighbour_colors = {colors[v] for v in G[u] if v in colors}
        for color in itertools.count():
            if color not in neighbour_colors:
                break
        colors[u] = color
    return colors


def DSatur(G, colors):
    distinct_colors = {v: set() for v in G}
    for i in range(len(G)):
        if i == 0:
            node = max(G, key=G.degree)
            yield node
            for v in G[node]:
                distinct_colors[v].add(0)
        else:
            saturation = {
                v: len(c) for v, c in distinct_colors.items() if v not in colors
            }
            node = max(saturation, key=lambda v: (saturation[v], G.degree(v)))
            yield node
            color = colors[node]
            for v in G[node]:
                distinct_colors[v].add(color)
                

def count_dict(s):
    shift_rand_df = (pd
                     .DataFrame(list(zip(s.keys(), s.values())))
                     .rename(columns={0:'shift', 1:'worker'})
                    )
    res_df_costs = res_df.groupby('worker').count()[['shift']].reset_index().sort_values(by='shift')
    scores_dict = dict(zip(res_df_costs['worker'], res_df_costs['shift']))
                     
    new_dict = {}
    for key in s.keys():
        new_dict[key] = [scores_dict[s[key]], s[key]]
                     
    scores_dict_asc = dict(sorted(new_dict.items(), key=lambda x: x[1]))
    
    scores_dict_desc = dict(sorted(new_dict.items(), key=lambda x: x[1], reverse=True))
    return scores_dict_asc, scores_dict_desc


def get_cost(di, norm=3, alpha=0.1, beta=0.6):
    shift_rand_df = (pd
                     .DataFrame(list(zip(di.keys(), di.values())))
                     .rename(columns={0:'shift', 1:'worker'})
                    )
    res_df_costs = shift_rand_df.groupby('worker').count()[['shift']].reset_index().sort_values(by='shift')
    scores_dict = dict(zip(res_df_costs['worker'], res_df_costs['shift']))
    cost = 0
    for key in scores_dict:
        if scores_dict[key] < norm:
            cost+=((norm-scores_dict[key])*alpha)
        else:
            cost+=((scores_dict[key]-norm)*beta)
    return cost
                     
            
def swap(low, high, s, G):
    nei_low = list(G.neighbors(low))
    nei_high = list(G.neighbors(high))
    colors_nei_low = [s[y] for y in nei_low]
    colors_nei_high = [s[y] for y in nei_high]
    if s[high] in colors_nei_low or s[low] in colors_nei_high:
        return False
    else:
        return True
    
    
def find_best_color(G, s, high, init_solution):
    nei = list(G.neighbors(high))
    colors_nei = []
    for key in nei:
        colors_nei.append(s[key])
    colors = list(range(max(init_solution.values())))
    C = list(set(colors) - set(colors_nei))
    if len(C) > 0:
        val = np.random.choice(C)
    else:
        val = -1
    return val


def count_dict(s):
    shift_rand_df = (pd
                     .DataFrame(list(zip(s.keys(), s.values())))
                     .rename(columns={0:'shift', 1:'worker'})
                    )
    res_df_costs = shift_rand_df.groupby('worker').count()[['shift']].reset_index().sort_values(by='shift')
    scores_dict = dict(zip(res_df_costs['worker'], res_df_costs['shift']))
                     
    new_dict = {}
    for key in s.keys():
        new_dict[key] = [scores_dict[s[key]], s[key]]
                     
    scores_dict_asc = dict(sorted(new_dict.items(), key=lambda x: x[1]))
    
    scores_dict_desc = dict(sorted(new_dict.items(), key=lambda x: x[1], reverse=True))
    return scores_dict_asc, scores_dict_desc

def get_cost(di, norm=3, alpha=0.1, beta=0.6):
    shift_rand_df = (pd
                     .DataFrame(list(zip(di.keys(), di.values())))
                     .rename(columns={0:'shift', 1:'worker'})
                    )
    res_df_costs = shift_rand_df.groupby('worker').count()[['shift']].reset_index().sort_values(by='shift')
    scores_dict = dict(zip(res_df_costs['worker'], res_df_costs['shift']))
    cost = 0
    for key in scores_dict:
        if scores_dict[key] < norm:
            cost+=((norm-scores_dict[key])*alpha)
        else:
            cost+=((scores_dict[key]-norm)*beta)
    return cost
                     
            
def swap(low, high, s, G):
    nei_low = list(G.neighbors(low))
    nei_high = list(G.neighbors(high))
    colors_nei_low = [s[y] for y in nei_low]
    colors_nei_high = [s[y] for y in nei_high]
    if s[high] in colors_nei_low or s[low] in colors_nei_high:
        return False
    else:
        return True
    
def find_best_color(G, s, high, init_solution):
    nei = list(G.neighbors(high))
    colors_nei = []
    for key in nei:
        colors_nei.append(s[key])
    colors = list(range(max(init_solution.values())))
    C = list(set(colors) - set(colors_nei))
    if len(C) > 0:
        val = np.random.choice(C)
    else:
        val = -1
    return val


def tabu_search(G, init_solution, adj_matrix, max_iterations=3):
    #using notation from paper
    s_0 = init_solution
    TL = []
    s = s_0
    best = s_0
    iterations = 0
    stats = []
    flags = []
    cost = []
    while iterations < max_iterations:
        flag=0
        it = 1
        asc_dict, desc_dict = count_dict(s)
        for low in tqdm(asc_dict.keys()):
            for high in desc_dict.keys():
                if [low, high] not in TL:
                    it+=1
                    s_star = deepcopy(s)
                    if [low, high] in adj_matrix and swap(low, high, s, G):
                        temp = s_star[low]
                        s_star[low] = s_star[high]
                        s_star[high] = temp
                        if get_cost(s_star) < get_cost(s):
                            print('+')
                            TL.append([low, high])
                            TL.append([high, low])
                            s = s_star
                            flag = 1
                            break
                    else:
                        color = find_best_color(G, s, high, init_solution)
                        if color > -1:
                            s_star[high] = color
#                             print(get_cost(s_star), get_cost(s))
                            if get_cost(s_star) < get_cost(s):
                                TL.append([low, high])
                                TL.append([high, low])
                                s = s_star
                                flag = 1
                                break
            
            if flag == 1:
                break
        stats.append(it)
        flags.append(flag)
        cost.append(get_cost(s))
        if max(flags[-1:]) == 0:
            return cost, stats, s
        iterations+=1
    return cost, stats, s
