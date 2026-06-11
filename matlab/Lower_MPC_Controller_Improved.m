function [P_opt, info] = Lower_MPC_Controller_Improved(P_ref, P_prev, P_req_total, P_nom, SOH_current, h2_coeffs)
    % Multi-step receding-horizon MPC allocator for the multi-stack PEMFC system.
    %
    % Decision variable:
    %   U = [dP(k|k); dP(k+1|k); ...; dP(k+Nc-1|k)]
    %
    % Prediction model:
    %   P(k+j|k) = P(k-1) + sum_{l=0}^{j} dP(k+l|k)
    %
    % Only the first optimized increment is applied. The remaining sequence is
    % discarded at the next sample, which is the standard MPC rolling mechanism.

    N = length(P_ref);
    P_ref = P_ref(:);
    P_prev = P_prev(:);
    SOH_current = max(0.001, SOH_current(:));

    a_h2 = h2_coeffs(1);
    b_h2 = h2_coeffs(2);

    active = P_ref > 1e-3;
    if ~any(active) || P_req_total <= 1e-6
        P_opt = zeros(N, 1);
        info = localInfo(1, 0, 0, 0, "zero_request", 0, 0, 0);
        return;
    end

    % Horizons. Nc=6 means the optimizer plans six future power increments and
    % applies only the first increment in the current sampling period.
    Nc = 6;
    Nu = N * Nc;

    % Weights. Tracking total fuel-cell power remains the dominant objective,
    % while reference following, SOH protection, hydrogen economy and ramp
    % smoothness shape the distribution among stacks.
    W_sum = 1800.0;
    W_ref = 1.5;
    W_delta = 9.0;
    W_h2 = 18.0;
    W_soh = 35.0;
    W_terminal = 4.0;

    % Health weighting: aged stacks are penalized more heavily, therefore the
    % optimizer naturally shifts more predicted load to healthier stacks.
    soh_mean = mean(SOH_current(active));
    aging_bias = max(0, soh_mean - SOH_current);
    health_penalty = 1 + W_soh * aging_bias;
    % Inactive stacks are not forced to zero immediately. They are assigned a
    % strong operating penalty and must ramp down through the same predicted
    % dynamics and ramp-rate constraints as active stacks.
    health_penalty(~active) = 300;

    % Linearized hydrogen rate around the previous operating point.
    h2_grad = max(0, 2 * a_h2 * max(P_prev, 0) + b_h2);

    % Prediction matrix: P_pred = P0 + S * U.
    S_time = tril(ones(Nc));
    S = kron(S_time, eye(N));
    P0 = repmat(P_prev, Nc, 1);
    P_ref_vec = repmat(P_ref, Nc, 1);
    P_req_vec = repmat(P_req_total, Nc, 1);

    % Sum-of-stack matrix for each prediction step.
    Csum = kron(eye(Nc), ones(1, N));

    % Stage weights grow slightly over the horizon so the controller does not
    % only fix the first sample and ignore future feasibility.
    stage_gain = linspace(1.0, 1.35, Nc)';
    Wstage = kron(diag(stage_gain), eye(N));
    Whealth = kron(diag(stage_gain), diag(health_penalty));
    Wh2diag = kron(diag(stage_gain), diag(max(h2_grad, 1e-6)));

    H = zeros(Nu, Nu);
    f = zeros(Nu, 1);

    % 1) Total power tracking across the prediction horizon.
    H = H + 2 * W_sum * (Csum * S)' * (Csum * S);
    f = f + 2 * W_sum * (Csum * S)' * (Csum * P0 - P_req_vec);

    % 2) Reference tracking from the SOH-aware state machine.
    H = H + 2 * W_ref * S' * Wstage * S;
    f = f + 2 * W_ref * S' * Wstage * (P0 - P_ref_vec);

    % 3) SOH-aware distribution penalty. This is a predicted operating cost,
    % not a single-step heuristic, because it acts on all future P(k+j|k).
    H = H + 2 * S' * Whealth * S;
    f = f + 2 * S' * Whealth * P0;

    % 4) Hydrogen economy, using a local quadratic/linear approximation.
    H = H + 2 * W_h2 * S' * Wh2diag * S;
    f = f + 2 * W_h2 * S' * Wh2diag * P0;

    % 5) Control increment smoothness.
    H = H + 2 * W_delta * eye(Nu);

    % 6) Terminal shaping: keep the final predicted point close to P_ref.
    E_terminal = zeros(N, Nu);
    E_terminal(:, (Nc-1)*N + (1:N)) = eye(N);
    S_terminal = E_terminal * S;
    P0_terminal = E_terminal * P0;
    H = H + 2 * W_terminal * (S_terminal' * S_terminal);
    f = f + 2 * W_terminal * S_terminal' * (P0_terminal - P_ref);

    H = (H + H') / 2 + 1e-8 * eye(Nu);

    % Increment bounds for every future move.
    dP_up = 0.12 * P_nom;
    dP_down = -0.12 * P_nom;
    LB = dP_down * ones(Nu, 1);
    UB = dP_up * ones(Nu, 1);

    for j = 1:Nc
        for i = 1:N
            idx = (j-1) * N + i;
            if ~active(i)
                % Non-selected stacks should ramp down, not rise again.
                UB(idx) = 0;
            end
        end
    end

    % Absolute predicted power constraints:
    % active stacks: [0.05, 1.10] P_nom.
    % inactive stacks: [0, P_prev] with UB(dP)<=0, so they can only ramp down
    % progressively instead of being forced to zero at the first prediction step.
    Pmin = zeros(Nc * N, 1);
    Pmax = zeros(Nc * N, 1);
    for j = 1:Nc
        for i = 1:N
            idx = (j-1) * N + i;
            if active(i)
                Pmin(idx) = 0.05 * P_nom;
                Pmax(idx) = 1.10 * P_nom;
            else
                Pmin(idx) = 0;
                Pmax(idx) = max(P_prev(i), 0);
            end
        end
    end

    Aineq = [S; -S];
    bineq = [Pmax - P0; -(Pmin - P0)];

    % Use a soft total-power target: if a sharp request is infeasible within
    % ramp limits, the residual is handled by the battery and logged upstream.
    opts = optimoptions('quadprog', 'Display', 'off');
    [U, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, [], [], LB, UB, [], opts);

    if isempty(U) || exitflag <= 0
        % Feasible emergency fallback: one-step projection inside the same
        % ramp and absolute power envelopes. This should be rare and is logged.
        dP0 = localFeasibleFirstMove(P_prev, P_ref, active, P_nom);
        message = "fallback_feasible_projection";
    else
        dP0 = U(1:N);
        message = string(output.message);
    end

    P_opt = P_prev + dP0;
    P_opt = max(P_opt, 0);
    P_opt(P_opt < 1e-3) = 0;

    P_pred = P0;
    if ~isempty(U) && exitflag > 0
        P_pred = P0 + S * U;
    end

    first_gap = sum(P_opt) - P_req_total;
    predicted_sum = Csum * P_pred;
    max_prediction_gap = max(abs(predicted_sum - P_req_total));
    info = localInfo(exitflag, first_gap, first_gap, max_prediction_gap, ...
        message, Nc, Nu, norm(dP0, 2));
end

function dP0 = localFeasibleFirstMove(P_prev, P_ref, active, P_nom)
    N = length(P_prev);
    dP0 = zeros(N, 1);
    for i = 1:N
        if active(i)
            target = min(1.10 * P_nom, max(0.05 * P_nom, P_ref(i)));
        else
            target = 0;
        end
        raw = target - P_prev(i);
        dP0(i) = min(0.12 * P_nom, max(-0.12 * P_nom, raw));
    end
end

function info = localInfo(exitflag, residual, tracking_gap, max_prediction_gap, message, Nc, Nu, first_move_norm)
    info = struct();
    info.exitflag = exitflag;
    info.residual = residual;
    info.tracking_gap = tracking_gap;
    info.max_prediction_gap = max_prediction_gap;
    info.message = message;
    info.Nc = Nc;
    info.Nu = Nu;
    info.first_move_norm = first_move_norm;
end
