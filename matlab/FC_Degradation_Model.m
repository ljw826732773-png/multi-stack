function SOH_new = FC_Degradation_Model(SOH_old, P_curr, P_prev, P_nom, dt)
    % 纯净物理模型 (底层系数对应文献实测数据)
    N = length(SOH_old);
    K_ss = 1.96e-5;    % 启停损耗系数
    K_trans = 5.93e-5; % 变载损耗系数
    
    SOH_new = zeros(N, 1);
    for i = 1:N
        % 变载惩罚 ΔPfc
        deg_t = K_trans * (abs(P_curr(i) - P_prev(i)) / P_nom);
        % 起停惩罚
        deg_s = K_ss * abs((P_curr(i)>1) - (P_prev(i)>1));
        % 稳态/待机损耗：低功率或停机电堆仍保留少量环境待机衰退，
        % 但不再与满功率运行电堆承担相同稳态衰退。
        load_factor = max(P_curr(i) / P_nom, 0.1);
        deg_v = 1.2e-6 * dt * load_factor; 
        
        SOH_new(i) = SOH_old(i) - (deg_t + deg_s + deg_v);
    end
    SOH_new = max(0, min(1, SOH_new));
end
