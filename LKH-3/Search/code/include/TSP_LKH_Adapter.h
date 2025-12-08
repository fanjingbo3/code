// --- START OF FILE TSP_LKH_Adapter.h ---
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath> 

// 定义 LKH 候选集的 K 值（Top-K）
const int LKH_CANDIDATE_NUM = 5; // 可以根据需要调整，通常 5-20 之间
const char* LKH_EXE_PATH = "./LKH"; // 确保你的目录下有编译好的 LKH 可执行文件

struct Neighbor {
    int id;
    double potential; // 改名，不再叫 weight
    
    bool operator>(const Neighbor& other) const {
        return potential > other.potential;
    }
};

// 1. 将当前的 Weight 矩阵转换为 LKH 的候选集文件 (.cand)
// LKH 格式: NodeId Count Neighbor1 Alpha1 Neighbor2 Alpha2 ...
// 注意：LKH 的节点索引从 1 开始
void Generate_LKH_Candidates(const char* filename) {
    FILE* fp = fopen(filename, "w");
    // ... 错误检查 ...
    fprintf(fp, "%d\n", Virtual_City_Num);

    // 确保 Alpha 和 Total_Simulation_Times 可用（它们定义在 TSP_IO.h 或 TSP_MCTS.h 中）
    // 如果报未定义，请确保 include 了相关头文件，或者使用 extern 声明
    // extern double Alpha;
    // extern int Total_Simulation_Times;
    // extern int** Chosen_Times;
    // extern double Avg_Weight; 
    
    // 如果 Avg_Weight 为 0 (第一轮)，避免除以 0
    double safe_avg_weight = (Avg_Weight > 1e-6) ? Avg_Weight : 1.0;

    for (int i = 0; i < Virtual_City_Num; ++i) {
        std::vector<Neighbor> neighbors;
        for (int j = 0; j < Virtual_City_Num; ++j) {
            if (i == j) continue;

            // --- 核心修改：使用 UCB 公式计算潜力值 ---
            
            // 1. 利用项 (Exploitation)
            double exploitation = Weight[i][j]; 
            // 如果你想严格复刻 MCTS，可以用 Weight[i][j] / safe_avg_weight;
            
            // 2. 探索项 (Exploration)
            // 只要这条边被探索次数少，这个值就会很大
            double exploration = 0.0;
            if (Total_Simulation_Times > 0) {
                 exploration = Alpha * sqrt( log((double)Total_Simulation_Times + 1) / (double)(Chosen_Times[i][j] + 1) );
            }
            
            // 3. 总分
            double potential_score = exploitation + exploration;
            
            neighbors.push_back({j, potential_score});
        }

        // 按 Potential 降序排序
        std::sort(neighbors.begin(), neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.potential > b.potential;
        });

        // 写入文件 (LKH 只认 Top-K，但现在 Top-K 里混入了“探索边”)
        fprintf(fp, "%d 0 %d", i + 1, LKH_CANDIDATE_NUM); 
        for (int k = 0; k < LKH_CANDIDATE_NUM && k < neighbors.size(); ++k) {
            // 这里 Alpha 值我们还是按 rank * 100 给，骗 LKH 优先选前面的
            fprintf(fp, " %d %d", neighbors[k].id + 1, k * 100); 
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "-1\n");
    fclose(fp);
}
// 2. 生成 TSP 数据文件 (供 LKH 读取坐标)
void Generate_LKH_TSP_File(const char* filename) {
    FILE* fp = fopen(filename, "w");
    fprintf(fp, "NAME: temp_problem\n");
    fprintf(fp, "TYPE: TSP\n");
    fprintf(fp, "DIMENSION: %d\n", Virtual_City_Num);
    fprintf(fp, "EDGE_WEIGHT_TYPE: EUC_2D\n");
    fprintf(fp, "NODE_COORD_SECTION\n");
    for (int i = 0; i < Virtual_City_Num; ++i) {
        // LKH 需要整数或浮点坐标，这里保持一致
        fprintf(fp, "%d %.2f %.2f\n", i + 1, Coordinate_X[i], Coordinate_Y[i]);
    }
    fprintf(fp, "EOF\n");
    fclose(fp);
}

// 3. 生成 LKH 参数文件 (.par)
// --- TSP_LKH_Adapter.h ---

void Generate_LKH_Par_File(const char* par_file, const char* tsp_file, const char* cand_file, const char* tour_file) {
    FILE* fp = fopen(par_file, "w");
    if (!fp) return;

    fprintf(fp, "PROBLEM_FILE = %s\n", tsp_file);
    fprintf(fp, "CANDIDATE_FILE = %s\n", cand_file);
    fprintf(fp, "TOUR_FILE = %s\n", tour_file);
    
    // 这里的参数是为了让 LKH 跑得快且不仅仅依赖我们的输入
    fprintf(fp, "RUNS = 1\n"); 
    fprintf(fp, "MAX_TRIALS = 3\n"); // 推荐保留 5 或 10，给它一点微调空间
    
    // --- 【修改这里：实现 GNN + 距离 的混合候选集】 ---
    
    // 1. 设置类型为 ALPHA
    // 含义：LKH 会基于 Alpha-Nearness (距离) 生成候选集，但会优先读取 CANDIDATE_FILE
    fprintf(fp, "CANDIDATE_SET_TYPE = ALPHA\n"); 

    // 2. 设置最大候选数量 > 你的 LKH_CANDIDATE_NUM
    // 假设你的 LKH_CANDIDATE_NUM 是 5 (即 .cand 文件里每个点只有 5 个邻居)
    // 这里设置为 10，意味着：
    // 前 5 个位置 = 你的 GNN/热力图推荐 (从文件读)
    // 后 5 个位置 = LKH 自己按距离算的最近邻居 (自动补全)
    fprintf(fp, "MAX_CANDIDATES = 6\n"); 
    
    // --------------------------------------------------

    fprintf(fp, "SEED = %d\n", rand());
    fprintf(fp, "TRACE_LEVEL = 0\n"); 
    
    fclose(fp);
}

// 4. 读取 LKH 生成的 Tour 文件并更新到 All_Node
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
        if (strstr(buffer, "-1")) break; // End of list
        if (strstr(buffer, "EOF")) break;

        if (reading_section) {
            int city_id;
            if (sscanf(buffer, "%d", &city_id) == 1) {
                tour_nodes.push_back(city_id - 1); // 转回 0-based
            }
        }
    }
    fclose(fp);

    if (tour_nodes.size() != Virtual_City_Num) return Inf_Cost;

    // 将 vector 转回你的链表结构 All_Node
    for (size_t i = 0; i < tour_nodes.size(); ++i) {
        int u = tour_nodes[i];
        int v = tour_nodes[(i + 1) % tour_nodes.size()];
        int pre = tour_nodes[(i - 1 + tour_nodes.size()) % tour_nodes.size()];

        All_Node[u].Next_City = v;
        All_Node[u].Pre_City = pre;
    }

    return Get_Solution_Total_Distance();
}

// 5. 核心：基于 LKH 的搜索函数
// 替代原本的 MCTS 仿真过程
Distance_Type Run_LKH_Guided_Search(int Inst_Index) {
    // A. 准备文件路径
    char tsp_file[50], cand_file[50], par_file[50], tour_file[50];
    sprintf(tsp_file, "temp_%d.tsp", Inst_Index);
    sprintf(cand_file, "temp_%d.cand", Inst_Index);
    sprintf(par_file, "temp_%d.par", Inst_Index);
    sprintf(tour_file, "temp_%d.tour", Inst_Index);

    // B. 生成基于当前 Heatmap (Weight) 的候选集
    Generate_LKH_Candidates(cand_file);
    Generate_LKH_TSP_File(tsp_file);
    Generate_LKH_Par_File(par_file, tsp_file, cand_file, tour_file);

    // C. 调用 LKH
    char command[200];
    // 使用 ./LKH 或者 LKH.exe，确保路径正确
    sprintf(command, "%s %s > /dev/null", LKH_EXE_PATH, par_file);
    int ret = system(command);

    // D. 读取结果
    Distance_Type lkh_dist = Read_LKH_Tour(tour_file);

    // 清理临时文件 (可选)
    // remove(tsp_file); remove(cand_file); remove(par_file); remove(tour_file);

    return lkh_dist;
}

// 6. 热力图更新逻辑 (Back Propagation)
// 对应图片中的公式 H' = H' + Beta * (L_old - L_new)
void Update_Heatmap_With_LKH_Result(Distance_Type Old_Distance, Distance_Type New_Distance) {
    // 无论解是否变好，只要 LKH 跑过这个路径，我们就认为这些边被“访问”了一次
    // (或者你可以只统计变好的解，但为了消减探索项，最好是只要尝试了就计数)
    
    // 全局计数器 +1
    Total_Simulation_Times++; 

    int cur = Start_City;
    int count = 0;
    do {
        int next = All_Node[cur].Next_City;
        
        // 1. 更新访问次数 (用于让探索项变小，避免死盯着这条边)
        Chosen_Times[cur][next]++;
        Chosen_Times[next][cur]++;
        
        // 2. 只有解变好了才更新 Weight (反向传播)
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