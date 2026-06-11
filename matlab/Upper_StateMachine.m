function P_ref = Upper_StateMachine(P_req_total, SOH_current, P_nom, P_prev)
    N_stack = length(SOH_current);
    P_ref = zeros(N_stack, 1);
    SOH_current = max(0.001, SOH_current);
    
    if P_req_total > 260, N_active = 4;
    elseif P_req_total > 160, N_active = 3;
    elseif P_req_total > 60, N_active = 2;
    else, N_active = 1; end
    
    % --- 抗抖振迟滞环机制 (Hysteresis) ---
    % 给当前正在运行的电堆一个 0.005 的“虚拟健康加分”
    % 这样在 SOH 趋同后，系统不会因为万分之一的微弱差距就频繁切机
    SOH_virtual = SOH_current;
    SOH_virtual(P_prev > 1) = SOH_virtual(P_prev > 1) + 0.005;
    
    [~, sort_idx] = sort(SOH_virtual, 'descend');
    active_idx = sort_idx(1:N_active);
    SOH_active = SOH_current(active_idx);
    
    % --- 收敛后的死区平滑 ---
    if (max(SOH_active) - min(SOH_active)) < 0.005 
        P_ref(active_idx) = P_req_total / N_active;
    else 
        W = (SOH_active.^3) / sum(SOH_active.^3);
        P_ref(active_idx) = P_req_total .* W;
    end
    
    for i = 1:length(active_idx)
        idx = active_idx(i);
        P_ref(idx) = max(0.05*P_nom, min(1.1*P_nom, P_ref(idx)));
    end
    P_active_sum = sum(P_ref(active_idx));
    if P_active_sum > 0, P_ref(active_idx) = P_ref(active_idx) * (P_req_total / P_active_sum); end
end