function pD = LaplaceInvP(N, tD, fun_handle)
    % Stehfest 数值拉普拉斯反演 (效率优化版)
    
    if mod(N, 2) ~= 0
        error('Stehfest 反演参数 N 必须为偶数 (通常取 8, 10, 12, 14)');
    end
    if any(tD <= 0)
        error('送入反演的无因次时间 tD 必须全部大于 0');
    end
    
    % 1. 预计算 Stehfest 权重 V (只需计算一次，大幅提升效率)
    V = zeros(1, N);
    for i = 1:N
        for k = fix((i+1)/2) : min(i, N/2)
            V(i) = V(i) + k^(N/2) * factorial(2*k) / ...
                (factorial(N/2-k) * factorial(k) * factorial(k-1) * factorial(i-k) * factorial(2*k-i));
        end
        V(i) = (-1)^(N/2+i) * V(i);
    end
    
    % 2. 批量执行反演计算
    pD = zeros(size(tD));
    for i = 1:N
        pD = pD + V(i) * fun_handle(log(2) * i ./ tD);
    end
    
    % 3. 最终换算
    pD = log(2) * pD ./ tD;
end