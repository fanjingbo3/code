#ifndef TSP_LKH_ADAPTER_H
#define TSP_LKH_ADAPTER_H

#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

// 定义 LKH 候选集的 K 值
const int LKH_CANDIDATE_NUM = 5; 
const char* LKH_EXE_PATH = "./LKH"; 

struct Neighbor {
    int id;
    double potential; 
    bool operator>(const Neighbor& other) const {
        return potential > other.potential;
    }
};

// --- 新增函数：根据 Weight (GNN热图) 生成贪心初始解 ---
// 逻辑：从 0 开始，每次选 Weight 最大的未访问邻居
void Generate_GNN_Greedy_Initial_Tour(const char* filename) {
    std::vector<int> tour;
    std::vector<bool> visited(Virtual_City_Num, false);
    
    int current_city = 0; // 假设从 0 开始
    tour.push_back(current_city);
    visited[current_city] = true;

    for (int step = 1; step < Virtual_City_Num; ++step) {
        int best_next = -1;
        double max_weight = -1.0;

        // 寻找 Weight 最大的未访问邻居
        for (int next = 0; next < Virtual_City_Num; ++next) {
            if (!visited[next]) {
                // 这里直接用 Weight，也就是 GNN 的输出
                if (Weight[current_city][next] > max_weight) {
                    max_weight = Weight[current_city][next];
                    best_next = next;
                }
            }
        }

        // 如果万一全都是 0 或者没找到（极少情况），就找第一个没访问的
        if (best_next == -1) {
            for (int next = 0; next < Virtual_City_Num; ++next) {
                if (!visited[next]) {
                    best_next = next;
                    break;
                }
            }
        }

        current_city = best_next;
        tour.push_back(current_city);
        visited[current_city] = true;
    }

    // 将初始解写入文件 (LKH TOUR 格式)
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "NAME: GNN_Greedy_Init\n");
    fprintf(fp, "TYPE: TOUR\n");
    fprintf(fp, "DIMENSION: %d\n", Virtual_City_Num);
    fprintf(fp, "TOUR_SECTION\n");
    for (int id : tour) {
        fprintf(fp, "%d\n", id + 1); // LKH 从 1 开始
    }
    fprintf(fp, "-1\n");
    fprintf(fp, "EOF\n");
    fclose(fp);
}

// 1. 生成候选集文件 (.cand)
void Generate_LKH_Candidates(const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "%d\n", Virtual_City_Num);

    for (int i = 0; i < Virtual_City_Num; ++i) {
        std::vector<Neighbor> neighbors;
        for (int j = 0; j < Virtual_City_Num; ++j) {
            if (i == j) continue;
            // 依然使用 Weight + MCTS 探索项来生成候选集，给 LKH 一点微调空间
            double exploitation = Weight[i][j]; 
            double exploration = 0.0;
            if (Total_Simulation_Times > 0) {
                 exploration = Alpha * sqrt( log((double)Total_Simulation_Times + 1) / (double)(Chosen_Times[i][j] + 1) );
            }
            neighbors.push_back({j, exploitation + exploration});
        }
        std::sort(neighbors.begin(), neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.potential > b.potential;
        });

        fprintf(fp, "%d 0 %d", i + 1, LKH_CANDIDATE_NUM); 
        for (int k = 0; k < LKH_CANDIDATE_NUM && k < neighbors.size(); ++k) {
            fprintf(fp, " %d %d", neighbors[k].id + 1, k * 100); 
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "-1\n");
    fclose(fp);
}

// 2. 生成 TSP 数据文件
void Generate_LKH_TSP_File(const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "NAME: temp_problem\n");
    fprintf(fp, "TYPE: TSP\n");
    fprintf(fp, "DIMENSION: %d\n", Virtual_City_Num);
    fprintf(fp, "EDGE_WEIGHT_TYPE: EUC_2D\n");
    fprintf(fp, "NODE_COORD_SECTION\n");
    for (int i = 0; i < Virtual_City_Num; ++i) {
        fprintf(fp, "%d %.2f %.2f\n", i + 1, Coordinate_X[i], Coordinate_Y[i]);
    }
    fprintf(fp, "EOF\n");
    fclose(fp);
}

// 3. 生成 LKH 参数文件 (.par) - 这里做了重大修改
void Generate_LKH_Par_File(const char* par_file, const char* tsp_file, const char* cand_file, const char* tour_file, const char* init_tour_file) {
    FILE* fp = fopen(par_file, "w");
    if (!fp) return;

    fprintf(fp, "PROBLEM_FILE = %s\n", tsp_file);
    fprintf(fp, "CANDIDATE_FILE = %s\n", cand_file);
    fprintf(fp, "TOUR_FILE = %s\n", tour_file);
    
    // --- 核心限制修改 ---
    
    // 1. 指定初始解文件！
    // 这样 LKH 就会直接从你的 GNN 贪心解开始，而不是自己随机生成
    fprintf(fp, "INITIAL_TOUR_FILE = %s\n", init_tour_file);

    // 2. 限制只跑 1 次 Trial (只优化一次)
    // 既然初始解是我们给的，就不需要多次重启了。让它基于当前解做一次下降即可。
    fprintf(fp, "MAX_TRIALS = 1\n"); //拆边
    fprintf(fp, "RUNS = 1\n");

    // 3. 严格限制候选集，不补全，不自动优化权重
    fprintf(fp, "CANDIDATE_SET_TYPE = ALPHA\n"); 
    fprintf(fp, "MAX_CANDIDATES = %d\n", LKH_CANDIDATE_NUM); 
    fprintf(fp, "SUBGRADIENT = NO\n"); // 禁止 LKH 偷偷优化权重

    // 4. (可选) 如果你甚至想限制它优化的力度，可以把 MOVE_TYPE 改成 3
    // fprintf(fp, "MOVE_TYPE = 3\n"); // 默认是 5，非常强。改成 3 就比较弱。

    // -------------------

    fprintf(fp, "SEED = %d\n", rand());
    fprintf(fp, "TRACE_LEVEL = 0\n"); 
    
    fclose(fp);
}

// 4. 读取 Tour
Distance_Type Read_LKH_Tour(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return Inf_Cost;

    char buffer[256];
    bool reading_section = false;
    std::vector<int> tour_nodes;

    while (fgets(buffer, sizeof(buffer), fp)) {
        if (strstr(buffer, "TOUR_SECTION")) {
            reading_section = true;
            continue;
        }
        if (strstr(buffer, "-1")) break; 
        if (strstr(buffer, "EOF")) break;

        if (reading_section) {
            int city_id;
            if (sscanf(buffer, "%d", &city_id) == 1) {
                tour_nodes.push_back(city_id - 1); 
            }
        }
    }
    fclose(fp);

    if (tour_nodes.size() != Virtual_City_Num) return Inf_Cost;

    for (size_t i = 0; i < tour_nodes.size(); ++i) {
        int u = tour_nodes[i];
        int v = tour_nodes[(i + 1) % tour_nodes.size()];
        int pre = tour_nodes[(i - 1 + tour_nodes.size()) % tour_nodes.size()];
        All_Node[u].Next_City = v;
        All_Node[u].Pre_City = pre;
    }
    return Get_Solution_Total_Distance();
}

// 5. 运行 LKH 引导搜索 (主函数)
Distance_Type Run_LKH_Guided_Search(int Inst_Index) {
    char tsp_file[50], cand_file[50], par_file[50], tour_file[50], init_tour_file[50];
    sprintf(tsp_file, "temp_%d.tsp", Inst_Index);
    sprintf(cand_file, "temp_%d.cand", Inst_Index);
    sprintf(par_file, "temp_%d.par", Inst_Index);
    sprintf(tour_file, "temp_%d.tour", Inst_Index);
    sprintf(init_tour_file, "temp_%d.init", Inst_Index); // 初始解文件名

    // 1. 生成热力图对应的候选集
    Generate_LKH_Candidates(cand_file);
    // 2. 生成 TSP 问题文件
    Generate_LKH_TSP_File(tsp_file);
    // 3. 【新步骤】生成 GNN 贪心初始解
    Generate_GNN_Greedy_Initial_Tour(init_tour_file);
    
    // 4. 生成参数文件 (传入初始解文件)
    Generate_LKH_Par_File(par_file, tsp_file, cand_file, tour_file, init_tour_file);

    // 5. 运行 LKH
    char command[200];
    sprintf(command, "%s %s > /dev/null", LKH_EXE_PATH, par_file);
    int ret = system(command);

    // 6. 读取结果
    Distance_Type lkh_dist = Read_LKH_Tour(tour_file);

    // 清理文件 (建议保留以便调试)
    remove(tsp_file); remove(cand_file); remove(par_file); 
    remove(tour_file); remove(init_tour_file);

    return lkh_dist;
}

// 6. 更新热力图
void Update_Heatmap_With_LKH_Result(Distance_Type Old_Distance, Distance_Type New_Distance) {
    Total_Simulation_Times++; 

    int cur = Start_City;
    int count = 0;
    do {
        int next = All_Node[cur].Next_City;
        Chosen_Times[cur][next]++;
        Chosen_Times[next][cur]++;
        
        if (New_Distance < Old_Distance) {
             double Delta = (double)(Old_Distance - New_Distance);
             double Increase = Beta * Delta / (double)Magnify_Rate; 
             Weight[cur][next] += Increase;
             Weight[next][cur] += Increase;
        }
        cur = next;
        count++;
    } while (cur != Start_City && count < Virtual_City_Num + 2);
}

#endif