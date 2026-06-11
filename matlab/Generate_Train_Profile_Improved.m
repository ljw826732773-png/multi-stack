function [t_out, P_req_total, v_ms] = Generate_Train_Profile_Improved(T_end, dt)
    % ADVISOR cycle reader with signed traction power.
    % Difference from Generate_Train_Profile.m:
    %   1) negative power is preserved for regenerative braking analysis;
    %   2) no artificial 20 kW idle floor is imposed on vehicle demand;
    %   3) only a symmetric safety cap is applied to avoid unrealistic spikes.

    cyc_mph = [];
    CYC_UDDS;
    v_u = cyc_mph(:, 2);

    cyc_mph = [];
    CYC_HWFET;
    v_h = cyc_mph(:, 2);

    v_combined_mph = [v_u; v_h; v_u];
    v_ms_all = v_combined_mph * 0.44704;

    t_out = 0:dt:T_end;
    v_ms = interp1(0:(length(v_ms_all)-1), v_ms_all, t_out, 'linear', 'extrap');
    v_ms = max(0, v_ms);

    % Equivalent 12-ton commercial vehicle longitudinal model.
    m_v = 12000;
    Af = 5.5;
    cd = 0.45;
    cr = 0.012;
    rho = 1.225;
    g = 9.81;
    eta_drive = 0.874;
    eta_regen = 0.70;

    a = [diff(v_ms) / dt, 0];
    P_req_total = zeros(1, length(t_out));

    for k = 1:length(t_out)
        F_roll = cr * m_v * g;
        F_aero = 0.5 * rho * Af * cd * v_ms(k)^2;
        F_inertia = m_v * a(k);
        F_tract = F_inertia + F_roll + F_aero;
        P_mech = F_tract * v_ms(k);

        if P_mech >= 0
            P_req_total(k) = (P_mech / eta_drive) / 1000;
        else
            P_req_total(k) = (P_mech * eta_regen) / 1000;
        end
    end

    % Keep realistic vehicle-side power bounds while preserving braking power.
    P_req_total = max(min(P_req_total, 380), -180);
end
