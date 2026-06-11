%% 多堆燃料电池无控制对照组
clear; clc; close all;

%% 1. 环境与参数初始化
T_end = 1200; dt = 1; N_stack = 4; P_nom = 100;
t_time = 0:dt:T_end;

% 生成列车总功率需求
[~, P_total_req] = Generate_Train_Profile(T_end, dt);

% 初始 SOH 设定 (存在显著差异)
SOH_init = [0.95; 0.88; 0.82; 0.75]; 
SOH_hist = zeros(N_stack, length(t_time));
SOH_hist(:, 1) = SOH_init;

% 功率记录矩阵初始化
P_hist = zeros(N_stack, length(t_time));
P_hist(:, 1) = (P_total_req(1)/N_stack) * ones(N_stack, 1);

%% 2. 仿真主循环 (Baseline: 无脑均分，无启停调度)
fprintf('正在运行独立对照组 (Baseline) 仿真...\n');
for k = 2:length(t_time)
    P_req = P_total_req(k-1);
    
    % 传统等额分配：所有电堆强制均分功率，无法停机
    P_curr = (P_req / N_stack) * ones(N_stack, 1);
    
    P_hist(:, k) = P_curr;
    
    % 调用衰退模型计算当前步的 SOH
    SOH_hist(:, k) = FC_Degradation_Model(SOH_hist(:, k-1), P_curr, P_hist(:, k-1), P_nom, dt);
end

%% 3. 独立对照组数据可视化
figure('Name', 'Baseline 对照组独立运行结果', 'Color', 'w', 'Position', [150 150 800 600]);

% 子图 1：SOH 演化
subplot(2,1,1);
plot(t_time, SOH_hist', 'LineWidth', 1.5);
title('平均分配策略 (Baseline) SOH演化：持续分化且无收敛', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('SOH'); grid on;
legend('Stack 1 (0.95)','Stack 2 (0.88)','Stack 3 (0.82)','Stack 4 (0.75)', 'Location', 'southwest');

% 提取并展示最终的极差数据
diff_init = max(SOH_init) - min(SOH_init);
diff_end = max(SOH_hist(:, end)) - min(SOH_hist(:, end));
text(T_end*0.6, 0.9, sprintf('初始极差: %.4f\n最终极差: %.4f', diff_init, diff_end), ...
    'BackgroundColor', 'w', 'EdgeColor', 'k', 'FontWeight', 'bold');

% 子图 2：功率分配堆叠
subplot(2,1,2);
area(t_time, P_hist', 'EdgeColor', 'none', 'FaceAlpha', 0.6);
hold on; 
plot(t_time, P_total_req, 'k--', 'LineWidth', 2);
title('Baseline 功率分配：死板均分，产生大量无效怠速 (无法触发关机)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('时间 (s)'); ylabel('功率 (kW)');
legend('Stack 1','Stack 2','Stack 3','Stack 4','总需求', 'Location', 'best');
grid on;