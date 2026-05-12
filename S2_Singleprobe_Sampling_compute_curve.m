function result = S2_Singleprobe_Sampling_compute_curve(base, val, use_cache, cache)
% Version: 2026-05-12 V15
%
%   1) 按 param_choice 将外部传入的敏感性参数 val 覆盖进默认参数包；
%   2) 完成工程单位 (mD, m, h) 到无因次变量体系的规范换算；
%   3) 采用 Method of Images (镜像法) 与格林函数，构造有限面积探针的双重积分核；
%   4) 在拉氏空间依据 Duhamel 褶积定理叠加表皮(Skin)和井筒储存(Wellbore Storage)效应；
%   5) 调用 Stehfest 数值反演算法得到时间域物理压差，并计算 Bourdet 平滑导数。

% p 是本条曲线的局部参数副本。对 p 的修改仅限当前曲线范围，绝不污染全局状态。
p = base;
M_user_override = [];
F_user_override = [];

% 路由派发：将传入的测试值 val 覆盖到对应的目标物理量上
switch p.param_choice
    case 1,  p.C      = val;
    case 2,  p.S      = val;
    case 3,  p.Kh     = val;
    case 4,  p.Kv     = val;
    case 5,  p.hw     = val;
    case 6,  p.As_in2 = val;
    case 7,  p.q_val_default = val;
    case 8,  p.mu     = val;
    case 9,  p.omega  = val;
    case 10, p.lambda = val;
    case 11, p.Ri     = val;
    case 12, p.Kh2    = val;
    case 13, p.Kv2    = val;
    case 14, M_user_override = val;  % 流度比 M 覆盖
    case 15, F_user_override = val;  % 扩散比 F 覆盖
    case 16, p.omega2  = val;
    case 17, p.lambda2 = val;
end

% 流量换算：将工程常用输入 m^3/d 统一转换为核心解析式所需的标准尺度 cm^3/s
p.q = p.q_val_default * 1e6 / 86400;

% ---------------- 鲁棒性与参数合法性硬检查 ----------------
% 防御性编程：在进入复杂贝塞尔/矩阵计算前尽早抛出异常，防止 NaN 或 Inf 导致的排查困难
validate_stehfest_v15(p.stehfest_N);
validate_positive_v15('垂向级数最少项 series_min_terms', p.series_min_terms);
validate_positive_v15('垂向级数最大项 series_max_terms', p.series_max_terms);
validate_positive_v15('垂向级数截断阈值 series_tol', p.series_tol);
validate_positive_v15('垂向级数连续小项数 series_small_count', p.series_small_count);
if p.series_max_terms < p.series_min_terms
    error('series_max_terms 必须大于或等于 series_min_terms');
end
validate_positive_v15('油藏厚度 h', p.h);
validate_positive_v15('探针面积 As', p.As_in2);
validate_positive_v15('产量 q', p.q_val_default);
validate_positive_v15('流体粘度 mu', p.mu);
validate_positive_v15('孔隙度 phi', p.phi);
validate_positive_v15('综合压缩系数 ct', p.ct);
validate_positive_v15('井储系数 C', p.C);
validate_dual_media_v15('内区/单区', p.omega, p.lambda);
if p.model_choice == 4
    validate_dual_media_v15('外区', p.omega2, p.lambda2);
end
% 探针必须在储层内部
if p.hw <= 0 || p.hw >= p.h
    error('探针位置必须完全位于储层内部 (即 hw 必须在 (0, h) 开区间内)');
end
if p.Kh <= 0 || p.Kv <= 0
    error('内区/原始地层渗透率必须为正数');
end
if p.model_choice >= 3
    if p.Kh2 <= 0 || p.Kv2 <= 0
        error('外区渗透率 Kh2、Kv2 必须为正数');
    end
    validate_positive_v15('复合半径 Ri', p.Ri);
end

% ---------------- 基础几何与单位换算 (无因次化核心) ----------------
% Ke 是基于坐标变换法则计算的"等效球形渗透率"(Spherical Equivalent Permeability)。
% 将各项异性介质拉伸/压缩，映射为各向同性介质求解。
Ke = (p.Kh^2 * p.Kv)^(1/3);
As = p.As_in2 * 2.54^2;      % 平方英寸转平方厘米
rp = sqrt(As / pi);          % 等效物理圆面积的绝对半径
h_cm = p.h * 100;
hw_cm = p.hw * 100;
Ri_cm = p.Ri * 100;
L = h_cm;                    % 以储层总厚度 h 作为特征长度 L 进行无因次化

% ==== 探针表皮效应 (Skin) 的双重物理等效逻辑 ====
% 正表皮 (S>0)：地层伤害，等效为无穷薄的附加阻力层（在 apply_wellbore 中体现附加压降）。
% 负表皮 (S<0)：压裂或酸化增产，等效为探针有效吸流面积的扩大。
% 注：负表皮基于指数等效(exp(-S))源于二维径向流。针对早期的三维半球形流，它是一个工程实用近似。
if p.S < 0
    rp_skin = rp * exp(-p.S);
    rp_upper = p.negative_skin_max_factor * rp; % 限制过度放大引起的数值崩溃
    if p.model_choice >= 3
        rp_upper = min(rp_upper, 0.8 * Ri_cm);  % 防止有效吸流半径穿越至复合外区边界
    end
    rp_effective = min(rp_skin, rp_upper);
    S_residual_effective = 0;                   % 负表皮已转化为半径扩容，附加压降置 0
else
    rp_effective = rp;
    S_residual_effective = p.S;                 % 留给杜哈梅叠加
end

% 几何干涉校验：探针加上等效扩容半径后，不能穿透储层的上下封闭边界
if (hw_cm - rp_effective) <= 0 || (hw_cm + rp_effective) >= h_cm
    error('探针有限面积超出储层上下边界，请检查 hw、As 设定或负表皮引起的有效半径扩张');
end

% 无因次井筒储存系数 C_D
% 其推导消除了所有工程单位造成的量纲（m3/MPa 与 1/MPa 在公式中自洽抵消）
C_D = (p.C / (2 * pi * p.phi * p.ct * p.h * (L/100)^2)) * (Ke / p.Kh);

% ==== 坐标轴向拉伸与流度/扩散系数定义 ====
% 根据各项异性比率，进行径向 (scale_r1) 和垂向 (scale_z1) 的坐标伸缩
scale_r1 = sqrt(Ke / p.Kh);
scale_z1 = sqrt(Ke / p.Kv);
hD_1 = h_cm * scale_z1 / L;               % 内区无因次厚度
alpha_1 = p.Kh * scale_r1 / p.mu;         % 内区拉伸后的等效传导系数

M = 1; F = 1; hD_2 = []; RiD_1 = []; RiD_2 = []; alpha_2 = [];

% 针对复合模型 (Model 3 & 4)，建立外围介质体系，并计算界面反射参数
if p.model_choice >= 3
    Ke2 = (p.Kh2^2 * p.Kv2)^(1/3);
    scale_r2 = sqrt(Ke2 / p.Kh2);
    scale_z2 = sqrt(Ke2 / p.Kv2);
    hD_2 = h_cm * scale_z2 / L;
    
    % 注意：界面半径 Ri 在内外区拉伸比例不同时，对应不同的无因次坐标
    RiD_1 = Ri_cm * scale_r1 / L;
    RiD_2 = Ri_cm * scale_r2 / L;
    alpha_2_calc = p.Kh2 * scale_r2 / p.mu;
    
    % 导压系数 (Hydraulic Diffusivity)
    eta_2 = Ke2 / (p.phi * p.mu * p.ct);
    eta_1 = Ke / (p.phi * p.mu * p.ct);
    
    % 基准流度比 (Mobility Ratio) 与扩散比 (Diffusivity Ratio)
    M_calc = alpha_1 / alpha_2_calc;
    F_calc = eta_1 / eta_2;
    
    % 处理流度/扩散比强制覆盖敏感性 (UI选项 14 & 15)
    if isempty(M_user_override)
        M = M_calc;
        alpha_2 = alpha_2_calc;
    else
        M = M_user_override;
        alpha_2 = alpha_1 / M; % 反向推导外区系数
    end
    if isempty(F_user_override)
        F = F_calc;
    else
        F = F_user_override;
    end
end

% ---------------- 有限面积探针空间离散 ----------------
% 将探针吸流面由理论"点源"扩展为"面源"，通过数值网格化离散避免早期时间由于源点重合带来的奇异性(Singularity)。
[y_probe_cm, z_probe_cm, w_probe] = build_probe_points_from_radius_v15( ...
    rp_effective, hw_cm, p.probe_shape, p.Np_probe_target);
probe_point_count = numel(w_probe);

% 面源二次安全校验
if min(z_probe_cm) <= 0 || max(z_probe_cm) >= h_cm
    error('探针有限面积离散点超出储层上下边界...');
end

% 构造 Receiver (响应点) - Source (源点) 全连接矩阵
% 总共产生 N*N 个相互作用点对，这是面源积分的核心，计算量由此项主导
[YR, YS] = ndgrid(y_probe_cm, y_probe_cm);
[ZR, ZS] = ndgrid(z_probe_cm, z_probe_cm);
[WR, WS] = ndgrid(w_probe, w_probe);
pair_w = WR(:) .* WS(:);               % 点对综合权重
pair_w = pair_w ./ sum(pair_w);        % 权重归一化

% 为消除距离为 0 的自身奇异性，赋予中心一个微小的等效元胞半径
cell_radius_cm = sqrt((pi * rp_effective^2 / probe_point_count) / pi);
rho_cm = sqrt((YR(:) - YS(:)).^2 + cell_radius_cm^2);

pair_rhoD0 = rho_cm * scale_r1 / L;    % 相对无因次径向距离向量
pair_zr_frac = ZR(:) / h_cm;           % Receiver Z 轴归一化位置
pair_zs_frac = ZS(:) / h_cm;           % Source Z 轴归一化位置

% ---------------- 垂向镜像级数预结算加速 (空间域) ----------------
% 依据上下封闭边界的特性，格林函数在 z 方向会展开成一系列余弦项 Cos(n*pi*z/h)。
% 这些余弦项仅与探针空间分布有关，与拉氏反演时间变量 s 绝对解耦。
% 提前在矩阵中预结算并缓存，可规避在反演内部被重复计算数万次。
z_cache = zeros(numel(pair_w), p.series_max_terms);
for ncache = 1:p.series_max_terms
    z_cache(:, ncache) = cos(ncache * pi * pair_zr_frac) .* ...
        cos(ncache * pi * pair_zs_frac);
end

% ---------------- 时间和压力反尺度还原配置 ----------------
% tD = Ke[mD]*t[s] / (101.325*phi*mu*ct[1/MPa]*L[cm]^2)
% 101.325 是用于平衡混合工程单位系统与物理方程的恒定乘子
C_t_sec = 101.325 * (L^2 * p.phi_ct * p.mu) / Ke;
tD = (p.t_hours * 3600) / C_t_sec;

% 压力换算标度：用于将无因次压力 pD 重新映射回实际物理量 (MPa)
pressure_scale = (101.325 / (2 * pi)) * p.q * p.mu / (p.Kh * h_cm);

% 生成静态缓存键，防止重复构建地层响应核矩阵
static_cache_key = make_static_cache_key_v15(p, rp_effective, hD_1, hD_2, ...
    RiD_1, RiD_2, alpha_1, alpha_2, pair_rhoD0, pair_zr_frac, pair_zs_frac, M, F);

% 根据顶层意图，挂载不同的理论核函数句柄
switch p.model_choice
    case 1, fun_mode = @singleprobe_mode1;
    case 2, fun_mode = @singleprobe_mode2;
    case 3, fun_mode = @singleprobe_mode3;
    case 4, fun_mode = @singleprobe_mode4;
end

% ---------------- Stehfest 反演与导数计算 ----------------
% 使用 Stehfest 算法，将拉普拉斯域 (s) 积分形式反解为真实时间域响应
pD_inv = LaplaceInvP(p.stehfest_N, tD, fun_mode);
Dp = pressure_scale * pD_inv;

% 基于相邻时间对数的 Bourdet 滑动平均算法计算压力导数 (工业标准)
derivative = Bourdet_Smooth(p.t_hours, Dp, 0.1);

% 组装返回结果与元数据
result.param_value = val;
result.t_hours = p.t_hours;
result.Dp = Dp;
result.derivative = derivative;
result.meta = p;
result.meta.q = p.q;
result.meta.M = M;
result.meta.F = F;
result.meta.L = L;
result.meta.rp_effective = rp_effective;
result.meta.probe_point_count = probe_point_count;
result.meta.pressure_scale = pressure_scale;

% ==================== 四大物理模型底层实现 ====================
    function pD_wf = singleprobe_mode1(s)
        % 模型1：均质体系。直接传递拉氏参数 s。
        pD_form = finite_probe_layered_source(s, s, hD_1, alpha_1, ...
            'none', [], [], [], [], []);
        pD_wf = apply_wellbore(s, pD_form);
    end

    function pD_wf = singleprobe_mode2(s)
        % 模型2：双孔介质。利用 Warren-Root 定理，将参数 s 替换为带窜流迟滞效应的 f(s)*s
        f_s = double_porosity_kernel_v15(s, p.omega, p.lambda);
        pD_form = finite_probe_layered_source(s, s .* f_s, hD_1, alpha_1, ...
            'none', [], [], [], [], []);
        pD_wf = apply_wellbore(s, pD_form);
    end

    function pD_wf = singleprobe_mode3(s)
        % 模型3：径向复合。向外围传递修正了扩散比 F 的传导响应 s*F。
        pD_form = finite_probe_layered_source(s, s, hD_1, alpha_1, ...
            'two', s .* F, hD_2, alpha_2, RiD_1, RiD_2);
        pD_wf = apply_wellbore(s, pD_form);
    end

    function pD_wf = singleprobe_mode4(s)
        % 模型4：复合双孔介质。内外区分别计算各自的窜流因子 f0, f1，再叠加扩散比修正。
        f0 = double_porosity_kernel_v15(s, p.omega, p.lambda);
        f1 = double_porosity_kernel_v15(s, p.omega2, p.lambda2);
        pD_form = finite_probe_layered_source(s, s .* f0, hD_1, alpha_1, ...
            'two', s .* F .* f1, hD_2, alpha_2, RiD_1, RiD_2);
        pD_wf = apply_wellbore(s, pD_form);
    end

% ==================== 探针解析解：点源叠加与边界镜像 ====================
    function pD_form = finite_probe_layered_source(s, u0, hD0, alpha0, layer_mode, u1, hD1, alpha1, R0D0, R0D1)
        % 此处实现了由拉氏空间基本点源解到有边界约束的面源积分的完整闭环。
        
        % Stehfest 算法往往将 s 作为一个数组传入，此处采用递归降维处理，将其标量化以便参与缓存池查询
        if numel(s) > 1
            pD_form = zeros(size(s));
            for ss = 1:numel(s)
                pD_form(ss) = finite_probe_layered_source(s(ss), u0(ss), hD0, alpha0, layer_mode, ...
                    scalar_at_v15(u1, ss), hD1, alpha1, R0D0, R0D1);
            end
            return;
        end

        % 缓存隔离机制：
        % 注意地层核(pD_form)未受井储(C)与表皮(S)污染。这使得在进行 C/S 敏感性分析时，
        % 可直接命中缓存的地层核，时间可减少 90% 以上。
        cache_key = '';
        if use_cache && ~isempty(cache)
            cache_key = sprintf('%s|s%.12g|u0%.12g|u1%.12g|mode%s', ...
                static_cache_key, s, u0, scalar_at_v15(u1, 1), layer_mode);
            if isKey(cache, cache_key)
                pD_form = cache(cache_key);
                return;
            end
        end

        % --- 主流动阶段 (n=0 零阶项) ---
        % 代表垂直方向完全均压或无限大体系时的二维纯径向流动
        g0_0 = sqrt(u0);
        switch layer_mode
            case 'none'
                A0 = 0; % 无反射边界
            case 'two'
                g1_0 = sqrt(u1);
                % 请求复合边界产生的压力连续反射波
                A0 = reflection_two_layer(g0_0, g1_0, R0D0, R0D1, alpha0, alpha1);
            otherwise
                error('未知的 layer_mode');
        end
        % 将面元基本解进行加权积分累计
        P_sum = area_average_kernel(g0_0, A0, 0);

        % --- 上下封闭边界约束阶段 (n>=1 余弦镜像级数) ---
        n = 1;
        consecutive_small = 0;
        
        % 动态截断策略：
        % 由于早期的半球流收敛极慢，通常需要几百项。
        % series_tol 是相对误差比。为确保平滑，要求连续 small_count 次项均低于误差阈值时才准许安全退出。
        while n <= p.series_max_terms && (n <= p.series_min_terms || consecutive_small < p.series_small_count)
            term_z0 = (n * pi / hD0)^2;
            g0 = sqrt(u0 + term_z0); % 引入厚度影响的修正虚频
            switch layer_mode
                case 'none'
                    A = 0;
                case 'two'
                    term_z1 = (n * pi / hD1)^2;
                    g1 = sqrt(u1 + term_z1);
                    A = reflection_two_layer(g0, g1, R0D0, R0D1, alpha0, alpha1);
                otherwise
                    error('未知的 layer_mode');
            end
            addsum = 2 * area_average_kernel(g0, A, n);
            P_sum = P_sum + addsum;
            
            % 相对收敛判定
            if abs(addsum) < p.series_tol * max(1, abs(P_sum))
                consecutive_small = consecutive_small + 1;
            else
                consecutive_small = 0;
            end
            n = n + 1;
        end
        
        % 根据拉普拉斯时间积分性质，对单位脉冲源产生的压力核再除以 s 得到阶跃响应(定流量开井)
        pD_form = P_sum ./ s;
        
        if use_cache && ~isempty(cache)
            cache(cache_key) = pD_form;
        end
    end

% ==================== 面积分与界面反射核心 ====================
    function val_out = area_average_kernel(g, A, n)
        % 针对所有面源-响应点组，求解修正贝塞尔解。
        % base_part: 基岩内部传播的原始压力波幅衰减，形式为零阶第二类修正贝塞尔函数 K0。
        % reflect_part: 来自径向边界向内反射的回波，形式为零阶第一类修正贝塞尔函数 I0。
        arg = pair_rhoD0 .* g;
        base_part = besselk(0, arg);
        if abs(A) > 0
            reflect_part = A .* besseli(0, arg);
        else
            reflect_part = 0;
        end
        % 注入预存的垂向镜像约束网格点
        if n == 0
            z_part = 1;
        else
            z_part = z_cache(:, n);
        end
        % 点积加权整合
        val_out = sum(pair_w .* (base_part + reflect_part) .* z_part);
    end

    function A = reflection_two_layer(g0, g1, R0D0, R0D1, alpha0, alpha1)
        % 依据内外层压力(P1=P2)与法向流量连续边界条件联立解得的反射系数A。
        % 异常处理：在早期极限或者大尺度距离下，g0、g1 极大，导致 I0、K0 极易发生数据溢出(Infinity/Zero)。
        % 对策：启用 besseli(..., 1) 与 besselk(..., 1) 指数缩放形式。
        % 原本的 besselk(x) 被替换为 besselk(x)*exp(x)。
        if isempty(g1)
            A = 0;
            return;
        end
        z0 = g0 * R0D0;
        z1 = g1 * R0D1;
        I0_0_s = besseli(0, z0, 1); I1_0_s = besseli(1, z0, 1);
        K0_0_s = besselk(0, z0, 1); K1_0_s = besselk(1, z0, 1);
        K0_1_s = besselk(0, z1, 1); K1_1_s = besselk(1, z1, 1);
        
        q_outer = K1_1_s ./ K0_1_s;
        num_s = alpha0 .* g0 .* K1_0_s - alpha1 .* g1 .* q_outer .* K0_0_s;
        den_s = alpha0 .* g0 .* I1_0_s + alpha1 .* g1 .* q_outer .* I0_0_s;
        
        % exp(-2 * real(z0)) 用于还原指数缩放带来的漂移
        A = exp(-2 * real(z0)) .* (num_s ./ den_s);
    end

% ==================== 井储与表皮杜哈梅卷积叠加 ====================
    function pD_wf = apply_wellbore(s, pD_form)
        % 将无限传导地层模型转换为真实井筒表现。
        % 1. 附加常数阻力 (机械表皮)
        pD_skin = pD_form + S_residual_effective ./ s;
        % 2. 加入物质平衡效应 (井筒积液/气压缩效应)，该公式来源于杜哈梅卷积的频域表现
        pD_wf = pD_skin ./ (1 + C_D .* s.^2 .* pD_skin);
    end
end

% -------------------- 工具函数区块 --------------------
function key = make_static_cache_key_v15(p, rp_effective, hD_1, hD_2, RiD_1, RiD_2, alpha_1, alpha_2, pair_rhoD0, pair_zr_frac, pair_zs_frac, M, F)
% 构建地层核哈希特征串。
% 设计精髓：剔除 C、q、以及正表皮 S 的影响（它们不干预偏微分方程的底层传导响应），
% 但负表皮因重塑了吸流口几何轮廓 (rp_effective)，被强制纳入了特征签名当中。
skin_key = 0;
if p.S < 0
    skin_key = p.S;
end
key = sprintf(['m%d|Kh%.12g|Kv%.12g|Kh2%.12g|Kv2%.12g|Ri%.12g|h%.12g|hw%.12g|', ...
    'As%.12g|Sg%.12g|om%.12g|la%.12g|om2%.12g|la2%.12g|M%.12g|F%.12g|', ...
    'rp%.12g|hD1%.12g|hD2%.12g|RiD1%.12g|RiD2%.12g|a1%.12g|a2%.12g|', ...
    'nr%d|zr%.12g|zs%.12g'], ...
    p.model_choice, p.Kh, p.Kv, p.Kh2, p.Kv2, p.Ri, p.h, p.hw, p.As_in2, skin_key, ...
    p.omega, p.lambda, p.omega2, p.lambda2, M, F, rp_effective, hD_1, scalar_at_v15(hD_2, 1), ...
    scalar_at_v15(RiD_1, 1), scalar_at_v15(RiD_2, 1), alpha_1, scalar_at_v15(alpha_2, 1), ...
    numel(pair_rhoD0), sum(pair_zr_frac), sum(pair_zs_frac));
end

function f_s = double_porosity_kernel_v15(s, omega_in, lambda_in)
% Warren-Root 拟稳态双孔介质基质-裂缝窜流传递函数
% omega (0~1) 控制了裂缝本身的压缩性容积; lambda 控制了流体从基质流向裂缝通道的补给迟滞程度
f_s = (omega_in .* (1 - omega_in) .* s + lambda_in) ./ ...
    ((1 - omega_in) .* s + lambda_in);
end

function [y_cm, z_cm, w] = build_probe_points_from_radius_v15(rp_cm, hw_center_cm, shape_name, Ntarget)
% 面源数值积分用的靶向撒点算法
% 当前采用的是最稳定的等权重笛卡尔网格中心撒布策略，用以剔除圆外的无效网格
switch lower(shape_name)
    case 'circle'
        ngrid = max(5, ceil(sqrt(Ntarget * 4 / pi)) + 2);
        yy = linspace(-rp_cm, rp_cm, ngrid);
        zz = linspace(-rp_cm, rp_cm, ngrid);
        [Y, Z] = meshgrid(yy, zz);
        % 利用勾股定理构造理论圆形掩码 (Mask)
        mask = (Y.^2 + Z.^2) <= rp_cm^2;
        y_cm = Y(mask);
        z_cm = hw_center_cm + Z(mask);
        y_cm = y_cm(:);
        z_cm = z_cm(:);
        % 所有被保留的积分格点均分权重
        w = ones(size(y_cm)) / numel(y_cm);
    otherwise
        error('当前 V15 仅正式支持 circle 探针形状；ellipse/rectangle 可按同一接口拓展。');
end
end

function out = scalar_at_v15(x, idx)
% 柔性向量读取器
% 若拉氏算子 s 被视作标量处理，或者复合区不存在 (模型1、2)，安全反馈其自身或控制空异常。
if isempty(x)
    out = [];
elseif isscalar(x)
    out = x;
else
    out = x(idx);
end
end

function validate_stehfest_v15(N)
% Stehfest 反演项数校验。
% N 必须为正偶数；过小精度不足，过大在双精度下容易放大舍入误差。
if ~(isscalar(N) && isfinite(N) && N == round(N) && mod(N, 2) == 0 && N >= 6 && N <= 18)
    error('stehfest_N 必须为 6~18 之间的正偶数，推荐 10、12 或 14');
end
end

function validate_positive_v15(name, x)
% 严防负值或非数进入传递系统，引起诸如虚实数域跳跃的灾难错误
if ~(isscalar(x) && isfinite(x) && x > 0)
    error('%s 必须为正实数', name);
end
end

function validate_dual_media_v15(region_name, omega_in, lambda_in)
% Warren-Root 模型物理定义校验器：储容比必定处于区间 (0,1)
if ~(isscalar(omega_in) && isfinite(omega_in) && omega_in > 0 && omega_in < 1)
    error('%s储容比 omega 必须存在于 (0, 1) 的开区间内', region_name);
end
validate_positive_v15([region_name '窜流系数 lambda'], lambda_in);
end