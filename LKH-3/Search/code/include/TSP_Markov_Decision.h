#include "TSP_LKH_Adapter.h" // 引入上面的头文件
// Jump to a new state by randomly generating a solution
void Jump_To_Random_State()
{
	Generate_Initial_Solution(); 	
}

Distance_Type Markov_Decision_Process(int Inst_Index)
{
    MCTS_Init(Inst_Index);             // 这里的初始化很重要，载入 Rec/Prior 到 Weight
    Generate_Initial_Solution();       // 生成初始解
    Local_Search_by_2Opt_Move();       // 先简单优化一下

    // 记录初始最优
    Current_Instance_Best_Distance = Get_Solution_Total_Distance();
    Store_Best_Solution();

    int jump = 0;

    // 主循环：迭代 T 次或者时间截止
    while(((double)clock()-Current_Instance_Begin_Time) /CLOCKS_PER_SEC < Param_T * Virtual_City_Num)
    {
        // 1. 记录搜索前的解质量 (L(s))
        Distance_Type Distance_Before_Search = Get_Solution_Total_Distance();

        // 2. 运行 LKH 搜索
        // 这个函数会生成候选集 -> 跑 LKH -> 更新 All_Node -> 返回新距离 L(s')
        Distance_Type LKH_Delta_Distance = Run_LKH_Guided_Search(Inst_Index);

        // 3. 反向传播更新热力图 (Heatmap Update)
        // 如果 LKH 找到了比 "这一轮开始前" 更好的解
        if (LKH_Delta_Distance < Distance_Before_Search) {
            // 注意：Run_LKH_Guided_Search 内部已经把 All_Node 更新为新解了
            // 所以我们可以直接遍历 All_Node 来强化边
            Update_Heatmap_With_LKH_Result(Distance_Before_Search, LKH_Delta_Distance);

            // 更新全局最优
            if (LKH_Delta_Distance < Current_Instance_Best_Distance) {
                Current_Instance_Best_Distance = LKH_Delta_Distance;
                Store_Best_Solution();
                printf("New Best found by LKH: %d\n", Current_Instance_Best_Distance);
            }
        }

        // 4. 重启机制 (可选，保持你原有的逻辑)
        // 如果陷入局部最优太久，可以 Jump
        /*
        if (restart) {
             // ... existing restart logic ...
             Jump_To_Random_State();
             Local_Search_by_2Opt_Move();
        }
        */

       // 由于 LKH 也是局部搜索，我们可以不断循环：
       // Heatmap -> Candidates -> LKH -> Better Solution -> Update Heatmap -> New Candidates ...
       // 这是一个自我增强的循环 (Self-Improvement Loop)

       jump++;
    }

    cout << "Iterations (LKH calls): " << jump << endl;

    Restore_Best_Solution();

    if(Check_Solution_Feasible())
        return Get_Solution_Total_Distance();
    else
        return Inf_Cost;
}



