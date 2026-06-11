% =========================================================
% MATLAB 脚本：读取真实工况数据并计算重卡需求功率
% 适用对象：12吨级重型商用车多堆燃料电池混合动力系统
% =========================================================
clear; clc; close all;

% 【全局字体设置】修复中文乱码，使用微软雅黑
set(0,'defaultAxesFontName', 'Microsoft YaHei');
set(0,'defaultTextFontName', 'Microsoft YaHei');

%% 1. 读取工况文件数据
% 这里直接调取你文件夹里的 C_WTVC.mat
% 如果你想换工况，直接把名字改成 'CHTC_HT.mat' 或 'UDDS.mat' 即可
fileName = 'C_WTVC.mat'; 
dataStruct = load(fileName);

% 动态提取数据（防止不同工况文件里的变量名不一致）
fields = fieldnames(dataStruct);
cycleData = dataStruct.(fields{1});

% 通常标准的工况矩阵第一列是时间(s)，第二列是车速(km/h)
t = cycleData(:, 1);          % 时间向量 [s]
v_kmh = cycleData(:, 2);      % 车速向量 [km/h]
v_mps = v_kmh / 3.6;          % 将车速转换为 [m/s] 以便计算

% 计算加速度 [m/s^2] (采用差分法)
dt = [diff(t); 1];            % 时间步长 (通常是1s)
a_mps2 = [diff(v_mps); 0] ./ dt; 

%% 2. 基于12吨级重型商用车的纵向动力学计算需求功率
% 以下参数严格对应你论文中的 表5-1
m = 12000;      % 满载总质量 [kg]
Af = 6.5;       % 迎风面积 [m^2]
Cd = 0.55;      % 空气阻力系数
f = 0.006;      % 滚动阻力系数
eta_t = 0.92;   % 机械传动效率
rho = 1.225;    % 空气密度 [kg/m^3]
g = 9.81;       % 重力加速度 [m/s^2]

% 计算各项行驶阻力 [N]
F_roll = m * g * f * cos(0);                        % 滚动阻力 (假设平路)
F_air = 0.5 * rho * Cd * Af * (v_mps.^2);           % 空气阻力
F_acc = m * a_mps2;                                 % 加速阻力

% 计算总驱动力需求 [N]
F_total = F_roll + F_air + F_acc;

% 计算轮端需求功率 [kW]
P_wheel = (F_total .* v_mps) / 1000;

% 考虑机械传动效率，计算整车电功率需求 [kW]
P_req = zeros(size(P_wheel));
for i = 1:length(P_wheel)
    if P_wheel(i) >= 0
        % 驱动状态：需求功率放大
        P_req(i) = P_wheel(i) / eta_t;
    else
        % 制动状态：能量回收（如果你的项目考虑了回收，效率作为乘数）
        P_req(i) = P_wheel(i) * eta_t; 
    end
end

%% 3. 绘制符合顶刊审美的双Y轴工况图
figure('Color', 'w', 'Units', 'pixels', 'Position', [100, 100, 800, 600]);

% --- 上半部分：车速曲线 ---
subplot(2,1,1);
plot(t, v_kmh, 'LineWidth', 1.5, 'Color', '#0072BD'); 
grid on;
set(gca, 'GridLineStyle', '--', 'GridColor', [0.8 0.8 0.8], 'LineWidth', 1);
xlim([0 max(t)]);
ylim([0 max(v_kmh)*1.2]); % Y轴上限留20%裕度
ylabel('车速 [km/h]', 'FontSize', 12, 'FontName', 'Microsoft YaHei');
title(['(a) ', strrep(fileName, '.mat', ''), ' 车速轨迹'], 'FontSize', 12, 'FontName', 'Microsoft YaHei', 'Interpreter', 'none');

% --- 下半部分：需求功率曲线 ---
subplot(2,1,2);
plot(t, zeros(size(t)), 'k-', 'LineWidth', 1); hold on; % 零功率基准线
plot(t, P_req, 'LineWidth', 1.2, 'Color', '#D95319'); 
grid on;
set(gca, 'GridLineStyle', '--', 'GridColor', [0.8 0.8 0.8], 'LineWidth', 1);
xlim([0 max(t)]);

% 动态设置功率图的Y轴范围
p_min = min(P_req);
p_max = max(P_req);
ylim([p_min*1.2, p_max*1.2]); 

xlabel('时间 [s]', 'FontSize', 12, 'FontName', 'Microsoft YaHei');
ylabel('需求功率 [kW]', 'FontSize', 12, 'FontName', 'Microsoft YaHei');
title('(b) 整车需求功率分布', 'FontSize', 12, 'FontName', 'Microsoft YaHei');

% 调整子图间距以适应导出
set(gcf, 'PaperPositionMode', 'auto');