function results = S2_Singleprobe_Sampling(model_choice, param_choice)
% Version: 2026-05-12 V15
% 功能：单探针取样压力试井解释模型 - 参数敏感性分析
clc; close all;

need_model_input = nargin < 1 || isempty(model_choice);
need_param_input = nargin < 2 || isempty(param_choice);

%% --- 1. 用户输入：选择模型 ---
if need_model_input
    disp('-----------------------------------------');
    disp('单探针取样压力试井解释模型');
    disp('1. 均质模型');
    disp('2. 双重孔隙介质模型');
    disp('3. 径向复合均质模型');
    disp('4. 径向复合双重介质模型');
    model_choice = input('请输入数字 (1-4): ');
end

switch model_choice
    case 1
        model_name_cn = '均质';
    case 2
        model_name_cn = '双重孔隙介质';
    case 3
        model_name_cn = '径向复合均质';
    case 4
        model_name_cn = '径向复合双重介质';
    otherwise
        error('模型选择无效：请输入 1-4');
end

%% --- 2. 运行与数值算法设置：与物性参数无关 ---
% 这些设置只影响计算流程、离散精度或绘图行为，不代表储层物性本身。
do_plot = true;                         % 是否绘图：true=默认绘图；false=只返回 results，不弹出图窗
t_hours = logspace(-10, 6, 120);         % 时间网格 [h]：10^-10 到 10^6 小时，共 120 个对数等距点
Np_probe_target = 25;                    % 探针面积积分目标点数；圆形掩码后实际点数可能略大于该值
stehfest_N = 12;                         % Stehfest 拉氏反演阶数；建议使用 10、12、14 等偶数
series_min_terms = 24;                   % 垂向镜像级数最少累加项数，避免极早期过早截断
series_max_terms = 400;                  % 垂向镜像级数最大项数，防止极端参数下死循环
series_tol = 1e-7;                       % 垂向镜像级数相对截断误差阈值
series_small_count = 3;                  % 连续 small 项数量达到该值后才允许截断，使导数曲线更平滑
use_cache = true;                        % 是否启用地层响应核缓存；敏感性分析时通常能明显加速
use_parallel = false;                    % 是否启用并行计算；默认 false，避免无并行工具箱时产生兼容性问题
negative_skin_max_factor = 20;           % 负表皮等效半径最大放大倍数，防止非物理过度扩张

%% --- 3. 物理与工程参数赋值：与模型参数有关 ---
% --- 通用流动物性参数 ---
Kh      = 1000;                          % 水平渗透率；单区模型为原始地层，复合模型为内区水平渗透率 [mD]
Kv      = 50;                            % 垂向渗透率；单区模型为原始地层，复合模型为内区垂向渗透率 [mD]
S       = 1;                             % 表皮系数 [无因次]
C       = 1e-7;                          % 物理井筒储存系数 [m^3/MPa] 

% --- 储层与流体基础物性参数 ---
phi     = 0.2;                           % 储层孔隙度 [小数/无因次]
ct      = 5e-4;                          % 综合压缩系数 [1/MPa]
mu      = 5;                             % 地层流体粘度 [cP]
phi_ct  = phi * ct;                      % 导压系数相关项 (孔隙度与压缩系数乘积) [1/MPa]

% --- 探针几何与井身结构参数 ---
As_in2  = 0.21;                          % 探针面积 [in^2]
h       = 20;                            % 油藏总厚度 [m]
hw      = 10;                            % 探针中心距底界绝对位置 [m]

% --- 有限面积探针设置 ---
probe_shape      = 'circle';             % 当前正式支持 circle
% Np_probe_target 已在第2节集中设置；提高到 49、81 可提升面积积分精度，但计算量约按 O(N^2) 增加。

% --- 测试工作制度 (产量) ---
q_val_default = 80;                      % 测试产/注量 [m^3/d]

% --- 双孔介质模型参数 ---
omega   = 1e-2;                          % 单区/内区储容比 [无因次]
lambda  = 1e-8;                          % 单区/内区窜流系数 [无因次]
omega2  = 1e-2;                          % 外区储容比，仅模型4使用 [无因次]
lambda2 = 1e-8;                          % 外区窜流系数，仅模型4使用 [无因次]

% --- 径向复合模型专属参数 (适用于模型3 & 4) ---
Kh2     = 1;                             % 外区水平渗透率 [mD]
Kv2     = 0.5;                           % 外区垂向渗透率 [mD]
Ri      = 1000.0;                        % 复合区界面绝对半径 [m]


%% --- 4. 用户输入：选择分析参数 ---
if need_param_input
    disp('-----------------------------------------');
    disp(['当前模型: ' model_name_cn]);
    disp('请选择要进行敏感性分析的参数:');
    disp('1.  井筒储存系数 (C)');
    disp('2.  表皮系数 (S)');
    if model_choice <= 2
        disp('3.  水平渗透率 (Kh)');
        disp('4.  垂向渗透率 (Kv)');
    else
        disp('3.  内区水平渗透率 (Kh1)');
        disp('4.  内区垂向渗透率 (Kv1)');
    end
    disp('5.  探针中心位置 (hw)');
    disp('6.  探针面积 (As)');
    disp('7.  产量 (q)');
    disp('8.  流体粘度 (mu)');
    if model_choice == 2
        disp('9.  储容比 (omega)');
        disp('10. 窜流系数 (lambda)');
    elseif model_choice == 4
        disp('9.  内区储容比 (omega1)');
        disp('10. 内区窜流系数 (lambda1)');
    end
    if model_choice == 3 || model_choice == 4
        disp('11. 复合半径 (Ri)');
        disp('12. 外区水平渗透率 (Kh2)');
        disp('13. 外区垂向渗透率 (Kv2)');
        disp('14. 流度比 (M)');
        disp('15. 扩散比 (F)');
    end
    if model_choice == 4
        disp('16. 外区储容比 (omega2)');
        disp('17. 外区窜流系数 (lambda2)');
    end
    param_choice = input('请输入参数对应的数字: ');
end

if ~ismember(param_choice, valid_params_for_model(model_choice))
    error('参数选择 %d 不适用于当前模型 %s', param_choice, model_name_cn);
end

% 中文名称与敏感性参数序列字典
switch param_choice
    case 1
        para_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6];           para_name_cn = '井筒储存系数'; para_name_sym = 'C';
    case 2
        para_list = [-3, -1, 0, 2, 5];                         para_name_cn = '残余表皮系数'; para_name_sym = 'S_{res}';
    case 3
        para_list = [100, 300, 500, 1000, 2000];
        if model_choice <= 2
            para_name_cn = '水平渗透率'; para_name_sym = 'K_h';
        else
            para_name_cn = '内区水平渗透率'; para_name_sym = 'K_{h1}';
        end
    case 4
        para_list = [5, 10, 20, 50, 100];
        if model_choice <= 2
            para_name_cn = '垂向渗透率'; para_name_sym = 'K_v';
        else
            para_name_cn = '内区垂向渗透率'; para_name_sym = 'K_{v1}';
        end
    case 5
        para_list = [1, 3, 5, 10, 15];                         para_name_cn = '探针中心位置'; para_name_sym = 'h_w';
    case 6
        para_list = [0.21, 0.69, 0.79, 2.32, 6.23];            para_name_cn = '探针面积'; para_name_sym = 'A_s';
    case 7
        para_list = [20, 50, 80, 100, 150];                    para_name_cn = '产量'; para_name_sym = 'q';
    case 8
        para_list = [0.5, 1, 2, 5, 10];                        para_name_cn = '流体粘度'; para_name_sym = '\mu';
    case 9
        para_list = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1];
        if model_choice == 4
            para_name_cn = '内区储容比'; para_name_sym = '\omega_1';
        else
            para_name_cn = '储容比'; para_name_sym = '\omega';
        end
    case 10
        para_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5];
        if model_choice == 4
            para_name_cn = '内区窜流系数'; para_name_sym = '\lambda_1';
        else
            para_name_cn = '窜流系数'; para_name_sym = '\lambda';
        end
    case 11
        para_list = [100, 200, 500, 1000, 2000];               para_name_cn = '复合半径'; para_name_sym = 'R_i';
    case 12
        para_list = [0.1, 0.5, 1, 5, 10];                      para_name_cn = '外区水平渗透率'; para_name_sym = 'K_{h2}';
    case 13
        para_list = [0.05, 0.1, 0.5, 2, 5];                    para_name_cn = '外区垂向渗透率'; para_name_sym = 'K_{v2}';
    case 14
        para_list = [0.01, 0.1, 1, 10, 100];                   para_name_cn = '流度比'; para_name_sym = 'M';
    case 15
        para_list = [0.01, 0.1, 1, 10, 100];                   para_name_cn = '扩散比'; para_name_sym = 'F';
    case 16
        para_list = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1];            para_name_cn = '外区储容比'; para_name_sym = '\omega_2';
    case 17
        para_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5];            para_name_cn = '外区窜流系数'; para_name_sym = '\lambda_2';
    otherwise
        error('无效');
end
clean_para_name = regexprep(para_name_cn, '[ (m)]', '');
file_name_str = sprintf('单探针取样测试_%s_%s', model_name_cn, clean_para_name);

%% --- 5. 循环计算与绘图 ---
h1 = gobjects(1, numel(para_list)); 
results = repmat(struct('param_value', [], 't_hours', [], 'Dp', [], 'derivative', [], 'meta', []), ...
    1, numel(para_list));

if do_plot
    figure('Color', 'w', 'Name', file_name_str, 'FileName', file_name_str, 'NumberTitle', 'off');
end

% 定义曲线颜色库 (高辨识度 7 色)
color_bank = [0 0 1; 1 0 0; 0 0.5 0; 0 0 0; 0.75 0 0.75; 0 0.6 0.6; 0.6 0.2 0];
num_lines = numel(para_list);
color_map = zeros(num_lines, 3);
for kidx = 1:num_lines
    color_map(kidx, :) = color_bank(mod(kidx - 1, size(color_bank, 1)) + 1, :);
end

base = struct();
base.model_choice = model_choice;
base.param_choice = param_choice;
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
base.stehfest_N = stehfest_N;
base.series_min_terms = series_min_terms;
base.series_max_terms = series_max_terms;
base.series_tol = series_tol;
base.series_small_count = series_small_count;

cache = [];
can_parallel = use_parallel && license('test', 'Distrib_Computing_Toolbox');
if can_parallel
    parfor i = 1:numel(para_list)
        results(i) = S2_Singleprobe_Sampling_V15_compute_curve(base, para_list(i), false, []);
    end
else
    if use_cache
        cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end
    if param_choice == 7
        first_result = S2_Singleprobe_Sampling_V15_compute_curve(base, para_list(1), use_cache, cache);
        results(1) = first_result;
        for i = 2:numel(para_list)
            scale_q = para_list(i) / para_list(1);
            results(i) = first_result;
            results(i).param_value = para_list(i);
            results(i).Dp = first_result.Dp .* scale_q;
            results(i).derivative = first_result.derivative .* scale_q;
            results(i).meta.q_val_default = para_list(i);
            results(i).meta.q = para_list(i) * 1e6 / 86400;
            results(i).meta.pressure_scale = first_result.meta.pressure_scale .* scale_q;
        end
    else
        for i = 1:numel(para_list)
            results(i) = S2_Singleprobe_Sampling_V15_compute_curve(base, para_list(i), use_cache, cache);
        end
    end
end

for i = 1:numel(results)
    if do_plot
        current_color = color_map(i,:);
        h1(i) = loglog(t_hours, results(i).Dp, 'LineWidth', 1.5, 'Color', current_color);
        hold on
        if i == numel(results)
            set(h1(i), 'Marker', 'square', 'MarkerSize', 4);
            loglog(t_hours, results(i).derivative, '--', 'Color', current_color, ...
                'LineWidth', 1.5, 'Marker', 'diamond', 'MarkerSize', 4);
        else
            loglog(t_hours, results(i).derivative, '--', 'Color', current_color, 'LineWidth', 1.5);
        end
    end
end

final_meta = results(end).meta;
Kh = final_meta.Kh; Kv = final_meta.Kv; S = final_meta.S; C = final_meta.C;
mu = final_meta.mu; As_in2 = final_meta.As_in2; h = final_meta.h; hw = final_meta.hw;
q_val_default = final_meta.q_val_default; omega = final_meta.omega; lambda = final_meta.lambda;
omega2 = final_meta.omega2; lambda2 = final_meta.lambda2; Kh2 = final_meta.Kh2; Kv2 = final_meta.Kv2;
Ri = final_meta.Ri; M = final_meta.M; F = final_meta.F; rp_effective = final_meta.rp_effective;
L = final_meta.L; probe_point_count = final_meta.probe_point_count;

%% --- 6. 图表修饰 ---
if do_plot
grid on; grid minor; axis tight;
yl = ylim; ylim([yl(1) * 0.2, yl(2) * 5]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'FontName', 'Microsoft YaHei');
xlabel('时间 t [h]', 'FontSize', 12, 'FontName', 'Microsoft YaHei');
ylabel('压力及导数 \Deltap, d(\Deltap)/d(lnt) [MPa]', 'FontSize', 12, 'FontName', 'Microsoft YaHei');
title({['单探针取样测试 ' model_name_cn]; [para_name_cn '敏感性分析']}, ...
    'Interpreter', 'none', 'FontSize', 24, 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');

%% --- 7. UI分离构建：物理符号图例 & 紧凑独立坐标系表格 ---
% ==================== 构建左上角图例 ====================
legend_str = cell(numel(para_list), 1);
for kidx = 1:numel(para_list)
    legend_str{kidx} = [para_name_sym ' = ' num2str(para_list(kidx))];
end
leg_main = legend(h1, legend_str, 'Location', 'northwest');
set(leg_main, 'FontName', 'Consolas', 'FontSize', 10, 'Interpreter', 'tex');

% ==================== 动态参数列表UI ====================
params = {};
add_p('厚度 h(m)', -1, h); 
add_p('产量 q(m3/d)', 7, q_val_default); 
if model_choice <= 2
    add_p('水平渗 Kh(mD)', 3, Kh);
    add_p('垂向渗 Kv(mD)', 4, Kv);
else
    add_p('内区水平渗 Kh1', 3, Kh);
    add_p('内区垂向渗 Kv1', 4, Kv);
end
add_p('残余表皮 S', 2, S); 
add_p('井储系数 C', 1, C);             
if model_choice >= 3
    add_p('复合界面 Ri(m)', 11, Ri);
    add_p('外区水平渗 Kh2', 12, Kh2);
    add_p('外区垂向渗 Kv2', 13, Kv2);
    add_p('流度比 M', 14, M);
    add_p('扩散比 F', 15, F);
end
if model_choice == 2 || model_choice == 4
    if model_choice == 4
        add_p('内区储容比 omega1', 9, omega);
        add_p('内区窜流 lambda1', 10, lambda);
        add_p('外区储容比 omega2', 16, omega2);
        add_p('外区窜流 lambda2', 17, lambda2);
    else
        add_p('储容比 omega', 9, omega);
        add_p('窜流系数 lambda', 10, lambda);
    end
end
add_p('探针位置 hw(m)', 5, hw);
add_p('探针面积 As(in2)', 6, As_in2); 
add_p('探针模型', -6, '有限面积');
add_p('有效半径 rp_eff(cm)', -2, rp_effective);
add_p('参考长度 L(cm)', -3, L);
add_p('探针离散点数', -4, probe_point_count);
add_p('粘度 mu(cP)', 8, mu);

% 表格引擎构建
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

% 调用外部独立交互Draggable.m
Draggable(ax_tab);
end


    function valid = valid_params_for_model(mdl)
        switch mdl
            case 1
                valid = 1:8;
            case 2
                valid = 1:10;
            case 3
                valid = [1:8, 11:15];
            case 4
                valid = 1:17;
            otherwise
                valid = [];
        end
    end

    function add_p(name, id, val)
        if id ~= param_choice
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
