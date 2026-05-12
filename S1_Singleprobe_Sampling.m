function result = S1_Singleprobe_Sampling(model_choice)
% Version: 2026-05-12 V15
% 功能：单探针取样压力试井解释模型 - 默认参数单曲线绘图
clc; close all;

% 不传入 model_choice 时，程序会在命令行提示选择 1~4 号物理模型。
need_model_input = nargin < 1 || isempty(model_choice);

%% --- 1. 用户交互：选择试井解释物理模型 ---
if need_model_input
    disp('-----------------------------------------');
    disp('单探针取样压力试井解释模型');
    disp('1. 均质模型');
    disp('2. 双重孔隙介质模型');
    disp('3. 径向复合单一介质模型');
    disp('4. 径向复合双重介质模型');
    model_choice = input('请输入数字 (1-4): ');
end

switch model_choice
    case 1
        model_name_cn = '均质';
    case 2
        model_name_cn = '双重孔隙介质';
    case 3
        model_name_cn = '径向复合单一介质';
    case 4
        model_name_cn = '径向复合双重介质';
    otherwise
        error('模型选择无效：请输入 1-4');
end

%% --- 2. 运行与数值算法设置：与物性参数无关 ---
% 本区块集中放置绘图、时间网格、积分精度、拉氏反演阶数、级数截断和缓存设置。
do_plot = true;                         % 是否绘图：true=默认绘图；false=只返回 result，不弹出图窗
t_hours = logspace(-10, 6, 120);         % 时间网格 [h]：10^-10 到 10^6 小时，共 120 个对数等距点
Np_probe_target = 25;                    % 探针面积积分目标点数；实际圆形掩码点数可能略大于该值
stehfest_N = 12;                         % Stehfest 拉氏反演阶数；建议使用 10、12、14 等偶数
series_min_terms = 24;                   % 垂向镜像级数最少累加项数，避免极早期过早截断
series_max_terms = 400;                  % 垂向镜像级数最大项数，防止极端参数下死循环
series_tol = 1e-7;                       % 垂向镜像级数相对截断误差阈值
series_small_count = 3;                  % 连续 small 项数量达到该值后才允许截断，使导数曲线更平滑
use_cache = true;                        % 是否启用地层响应核缓存；单曲线收益有限，但保持与 S2 逻辑一致
negative_skin_max_factor = 20;           % 负表皮等效半径最大放大倍数，防止非物理过度扩张

%% --- 3. 物理与工程参数赋值：与模型参数有关 ---
% === 流动物性参数 ===
Kh      = 1000;                          % 水平渗透率 [mD] (单区为原始地层，复合为内区)
Kv      = 10;                            % 垂向渗透率 [mD] (用于体现储层各项异性 Ke=(Kh^2*Kv)^(1/3))
S       = 1;                             % 机械表皮系数 [无因次] (探针周围的污染或压实效应)
C       = 1e-7;                          % 物理井筒储存系数 [m^3/MPa] (探针内容积流体压缩导致的早期效应)

% === 储层与流体基础物性 ===
phi     = 0.08;                           % 孔隙度 [小数/无因次]
ct      = 5e-4;                          % 综合压缩系数 [1/MPa] (岩石骨架压缩 + 流体压缩)
mu      = 5;                             % 地层流体粘度 [cP]
phi_ct  = phi * ct;                      % 导压系数相关项 [1/MPa]

% === 探针几何与位置参数 ===
As_in2  = 0.21;                          % 探针吸入口物理面积 [in^2]
h       = 20;                            % 油藏有效厚度 [m]
hw      = 10;                            % 探针中心距底界的绝对垂直距离 [m]

probe_shape      = 'circle';             % 探针形状：圆形最符合实际 MDT 探针物理特性

% 探针面离散点目标数 Np_probe_target 已在第2节设置。
% 数值越大，有限面积格林函数积分越精细，早期拟合精度越高，但计算量约按 O(N^2) 增加。
q_val_default = 80;                      % 取样测试的产液量/注水量 [m^3/d]

% === 双孔介质特征参数 (Warren-Root 模型) ===
omega   = 1e-2;                          % 储容比：裂缝系统储藏流体的能力占比 [无因次]
lambda  = 1e-4;                          % 窜流系数：基质向裂缝供给流体的能力强弱 [无因次]
omega2  = 1e-3;                          % 外区储容比，仅针对模型4 [无因次]
lambda2 = 1e-9;                          % 外区窜流系数，仅针对模型4 [无因次]

% === 径向复合边界参数 (适用于模型3、4) ===
Kh2     = 10;                             % 外区水平渗透率 [mD] (如未被泥浆滤液污染的原状地层)
Kv2     = 0.5;                           % 外区垂向渗透率 [mD]
Ri      = 1000.0;                        % 复合区界面(内外区交界)的绝对径向距离 [m]

% 构造统一参数包 base，向底层 worker 传递
% S1 模式不直接实现模型核，以此保证 S1/S2 两套代码的底层物理公式绝对一致，无漂移风险
base = struct();
base.model_choice = model_choice;
base.param_choice = 0;                   % S1 模式下不做参数敏感性替换，worker 将保持默认参数
base.t_hours = t_hours;
base.Kh = Kh; base.Kv = Kv; base.S = S; base.C = C;
base.phi = phi; base.ct = ct; base.mu = mu; base.phi_ct = phi_ct;
base.As_in2 = As_in2; base.h = h; base.hw = hw;
base.probe_shape = probe_shape; base.Np_probe_target = Np_probe_target;
base.negative_skin_max_factor = negative_skin_max_factor;
base.q_val_default = q_val_default;
base.omega = omega; base.lambda = lambda;
base.omega2 = omega2; base.lambda2 = lambda2;
base.Kh2 = Kh2; base.Kv2 = Kv2; base.Ri = Ri;

% === 数值算法控制参数 ===
% 调参建议：追求计算速度时可降低 N 和截断项数；输出论文级高精度图版时应适当提高
base.stehfest_N = stehfest_N;               % Stehfest 算法反演项数
base.series_min_terms = series_min_terms;   % 垂向镜像级数强行保证的最少累加项数
base.series_max_terms = series_max_terms;   % 防止极端参数下级数不收敛导致的死循环
base.series_tol = series_tol;               % 级数截断的相对误差阈值
base.series_small_count = series_small_count; % 连续出现微小项的次数，用于平滑截断

% 内存缓存机制：由于 S1 只有一条曲线，收益不如 S2，但保留此机制以统一接口
cache = [];
if use_cache
    cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
end

% 调用底层解析解计算模块
% val=0 是占位值。因 base.param_choice=0，worker 不会触发任何 override
result = S2_Singleprobe_Sampling_V15_compute_curve(base, 0, use_cache, cache);
result.param_value = [];

%% --- 4. 图版绘制与 UI 构建 ---
if do_plot
    % 生成高标准专业双对数试井图版
    file_name_str = sprintf('单探针取样测试_%s_默认参数_V15', model_name_cn);
    figure('Color', 'w', 'Name', file_name_str, 'FileName', file_name_str, 'NumberTitle', 'off');
    
    % 绘制压差曲线 (实线)
    loglog(t_hours, result.Dp, 'b-', 'LineWidth', 1.5);
    hold on;
    % 绘制 Bourdet 压力导数曲线 (虚线)
    loglog(t_hours, result.derivative, 'r--', 'LineWidth', 1.5);
    
    grid on; grid minor; axis tight;
    
    % 优化 Y 轴留白比例，防止早期导数下探被截断
    yl = ylim; ylim([yl(1) * 0.2, yl(2) * 5]);
    
    set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'FontName', 'Microsoft YaHei');
    xlabel('时间 t [h]', 'FontSize', 12, 'FontName', 'Microsoft YaHei');
    ylabel('压力及导数 \Deltap, d(\Deltap)/d(lnt) [MPa]', 'FontSize', 12, 'FontName', 'Microsoft YaHei');
    title(['单探针取样测试 ' model_name_cn ' 默认参数'], ...
        'Interpreter', 'none', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
    legend({'压力', '压力导数'}, 'Location', 'northwest', 'FontName', 'Microsoft YaHei');

    % 在图窗右下角渲染参数浮动列表
    build_parameter_table(result.meta);
end

    % ----------------- 内部辅助函数 -----------------
    function build_parameter_table(meta)
        % 动态构建 UI 参数信息卡片，复用 S2 风格
        params = {};
        add_p('厚度 h(m)', meta.h);
        add_p('产量 q(m3/d)', meta.q_val_default);
        if model_choice <= 2
            add_p('水平渗 Kh(mD)', meta.Kh);
            add_p('垂向渗 Kv(mD)', meta.Kv);
        else
            add_p('内区水平渗 Kh1', meta.Kh);
            add_p('内区垂向渗 Kv1', meta.Kv);
            add_p('复合界面 Ri(m)', meta.Ri);
            add_p('外区水平渗 Kh2', meta.Kh2);
            add_p('外区垂向渗 Kv2', meta.Kv2);
            add_p('流度比 M', meta.M);
            add_p('扩散比 F', meta.F);
        end
        add_p('残余表皮 S', meta.S);
        add_p('井储系数 C', meta.C);
        if model_choice == 2
            add_p('储容比 omega', meta.omega);
            add_p('窜流系数 lambda', meta.lambda);
        elseif model_choice == 4
            add_p('内区储容比 omega1', meta.omega);
            add_p('内区窜流 lambda1', meta.lambda);
            add_p('外区储容比 omega2', meta.omega2);
            add_p('外区窜流 lambda2', meta.lambda2);
        end
        add_p('探针位置 hw(m)', meta.hw);
        add_p('探针面积 As(in2)', meta.As_in2);
        add_p('探针模型', '有限面积');
        add_p('有效半径 rp_eff(cm)', meta.rp_effective);
        add_p('参考长度 L(cm)', meta.L);
        add_p('探针离散点数', meta.probe_point_count);
        add_p('粘度 mu(cP)', meta.mu);

        % 以下为 UI 坐标计算与绘制逻辑 (原生 axes 模拟表格)
        num_params = numel(params);
        num_rows = ceil(num_params / 2);
        left_params  = params(1:num_rows);
        right_params = params(num_rows+1:end);
        row_h   = 0.022;
        title_h = 0.035;
        box_w   = 0.36;
        box_h   = num_rows * row_h + title_h;
        ax_tab = axes('Position', [0.15, 0.13, box_w, box_h], 'Color', 'w', ...
            'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.8);
        ax_tab.XLim = [0, 1]; ax_tab.YLim = [0, num_rows + 1];
        y_title = num_rows + 0.5; y_line  = num_rows;
        text(ax_tab, 0.5, y_title, '参数列表', 'HorizontalAlignment', 'center', ...
            'FontName', 'Microsoft YaHei', 'FontSize', 9, 'FontWeight', 'bold');
        line(ax_tab, [0 1], [y_line y_line], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 0.5);
        line(ax_tab, [0.5 0.5], [0 y_line], 'Color', 'k', 'LineStyle', '-', 'LineWidth', 0.8);
        for r = 1:num_rows
            y_pos = num_rows - r + 0.5;
            p1 = left_params{r};
            text(ax_tab, 0.02, y_pos, p1{1}, 'HorizontalAlignment', 'left', 'FontName', 'Consolas', 'FontSize', 8);
            text(ax_tab, 0.34, y_pos, p1{2}, 'HorizontalAlignment', 'left', 'FontName', 'Consolas', 'FontSize', 8);
            if r <= numel(right_params)
                p2 = right_params{r};
                text(ax_tab, 0.52, y_pos, p2{1}, 'HorizontalAlignment', 'left', 'FontName', 'Consolas', 'FontSize', 8);
                text(ax_tab, 0.84, y_pos, p2{2}, 'HorizontalAlignment', 'left', 'FontName', 'Consolas', 'FontSize', 8);
            end
        end
        set(get(ax_tab, 'Children'), 'HitTest', 'off');
        % 调用外部实现的可拖拽 UI 组件
        Draggable(ax_tab);

        function add_p(name, val)
            % 统一参数值显示格式：针对极小或极大的科学计数法自动适配
            if isempty(val)
                v_str = '-';
            elseif ischar(val) || isstring(val)
                v_str = char(val);
            elseif abs(val) > 0 && (abs(val) < 1e-3 || abs(val) >= 1e4)
                v_str = sprintf('%g', val);
            else
                v_str = num2str(val);
            end
            params{end+1} = {name, v_str};
        end
    end
end