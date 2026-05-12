% 1_make_data
% 生成 4 参数反演专用的均质理论数据。
%
% 运行方式：
%   直接点击 MATLAB 编辑器的 Run，或在命令行执行：
%   run('1_make_data.m')
%
% 主要输出：
%   data_2000.mat
%
% 说明：
%   本脚本只随机采样 Kh、Kv、S、C。
%   q、mu、As、hw 作为已知参数固定。
%   文件中的 X 仍保存 8 列，方便记录已知参数；神经网络只读取前 4 列。

N = 2000;
out_file = 'data_2000.mat';
opts = struct();

dataset = make_data_impl(N, out_file, opts);

function dataset = make_data_impl(N, out_file, opts)

if nargin < 1 || isempty(N)
    N = 2000;
end
if nargin < 2 || isempty(out_file)
    out_file = 'data_2000.mat';
end
if nargin < 3 || isempty(opts)
    opts = struct();
end

% 固定随机种子，保证论文中的训练数据可复现。
rng(get_option(opts, 'seed', 20260507), 'twister');

% 新训练集覆盖 C 敏感的更早期时间段，同时保留晚期平台。
t_hours = get_option(opts, 't_hours', logspace(-8, 6, 120));
Nt = numel(t_hours);

base = make_base_template(t_hours, opts);

param_names = {'Kh_mD','Kv_mD','S','C_m3_per_MPa','q_m3_per_d','mu_cP','As_in2','hw_m'};
P = numel(param_names);
X = zeros(N, P);
Y = nan(N, Nt);
D = nan(N, Nt);
status = strings(N, 1);
message = strings(N, 1);

fprintf('Generating %d homogeneous 4-parameter samples...\n', N);
fprintf('Fixed known parameters: q=%.6g m3/d, mu=%.6g cP, As=%.6g in2, hw=%.6g m\n', ...
    base.q_val_default, base.mu, base.As_in2, base.hw);
tic;

samples = repmat(base, N, 1);
for i = 1:N
    sample = sample_four_inversion_params(base);
    samples(i) = sample;
    X(i, :) = [sample.Kh, sample.Kv, sample.S, sample.C, ...
        sample.q_val_default, sample.mu, sample.As_in2, sample.hw];
end

use_parallel = get_option(opts, 'use_parallel', true) && license('test', 'Distrib_Computing_Toolbox');
if use_parallel
    try
        pool = gcp('nocreate');
        if isempty(pool)
            parpool;
        end
        parfor i = 1:N
            [Y(i, :), D(i, :), status(i), message(i)] = run_one_sample(samples(i));
        end
    catch ME
        warning(ME.identifier, '并行生成失败，改用串行生成。原因：%s', ME.message);
        for i = 1:N
            [Y(i, :), D(i, :), status(i), message(i)] = run_one_sample(samples(i));
            print_progress(i, N);
        end
    end
else
    for i = 1:N
        [Y(i, :), D(i, :), status(i), message(i)] = run_one_sample(samples(i));
        print_progress(i, N);
    end
end

elapsed_sec = toc;
ok_mask = status == "ok";

fixed_known = struct();
fixed_known.q_m3_per_d = base.q_val_default;
fixed_known.mu_cP = base.mu;
fixed_known.As_in2 = base.As_in2;
fixed_known.hw_m = base.hw;

meta = struct();
meta.model_choice = 1;
meta.model_name = 'homogeneous_4param_fixed_known';
meta.param_names = param_names;
meta.inverted_param_names = param_names(1:4);
meta.known_param_names = param_names(5:8);
meta.fixed_known = fixed_known;
meta.N = N;
meta.Nt = Nt;
meta.elapsed_sec = elapsed_sec;
meta.ok_count = nnz(ok_mask);
meta.fail_count = N - nnz(ok_mask);
meta.time_created = char(datetime('now'));
meta.ranges = parameter_ranges();
meta.numeric_options = struct( ...
    'stehfest_N', base.stehfest_N, ...
    'series_min_terms', base.series_min_terms, ...
    'series_max_terms', base.series_max_terms, ...
    'series_tol', base.series_tol, ...
    'series_small_count', base.series_small_count, ...
    'Np_probe_target', base.Np_probe_target);

dataset = struct();
dataset.X = X;
dataset.Y = Y;
dataset.D = D;
dataset.t_hours = t_hours;
dataset.param_names = param_names;
dataset.status = status;
dataset.message = message;
dataset.meta = meta;

save(out_file, 'X', 'Y', 'D', 't_hours', 'param_names', 'status', 'message', 'meta', '-v7.3');
fprintf('\nSaved: %s\n', out_file);
fprintf('OK: %d / %d, elapsed: %.2f s\n', meta.ok_count, N, elapsed_sec);

end

function base = make_base_template(t_hours, opts)
% 构造与 MATLAB 正演内核兼容的参数包。
base = struct();
base.model_choice = 1;
base.param_choice = 0;
base.t_hours = t_hours;

% 储层和探针基础参数。
base.phi = 0.2;
base.ct = 5e-4;
base.phi_ct = base.phi * base.ct;
base.h = 20;
base.probe_shape = 'circle';
base.negative_skin_max_factor = 20;

% 均质模型不使用这些参数，但 worker 需要字段存在。
base.omega = 1e-2;
base.lambda = 1e-8;
base.omega2 = 1e-2;
base.lambda2 = 1e-8;
base.Kh2 = 1;
base.Kv2 = 0.5;
base.Ri = 1000;

% 数值设置。最终论文数据如需更精细，可在 opts 中提高这些值。
base.Np_probe_target = get_option(opts, 'Np_probe_target', 5);
base.stehfest_N = get_option(opts, 'stehfest_N', 10);
base.series_min_terms = get_option(opts, 'series_min_terms', 10);
base.series_max_terms = get_option(opts, 'series_max_terms', 80);
base.series_tol = get_option(opts, 'series_tol', 1e-6);
base.series_small_count = get_option(opts, 'series_small_count', 3);

% 4 个反演参数的初始占位值，会在采样时覆盖。
base.Kh = 1000;
base.Kv = 50;
base.S = 1;
base.C = 1e-7;

% 4 个已知参数固定，不参与采样。
base.q_val_default = get_option(opts, 'q_m3_per_d', 80);
base.mu = get_option(opts, 'mu_cP', 5);
base.As_in2 = get_option(opts, 'As_in2', 0.21);
base.hw = get_option(opts, 'hw_m', 10);
end

function sample = sample_four_inversion_params(base)
% 只随机采样需要反演的 4 个参数。
r = parameter_ranges();
sample = base;
sample.Kh = log_uniform(r.Kh_mD);
sample.Kv = log_uniform(r.Kv_mD);
sample.S = uniform_range(r.S);
sample.C = log_uniform(r.C_m3_per_MPa);
end

function ranges = parameter_ranges()
% 与 Python 反演边界保持一致。
ranges = struct();
ranges.Kh_mD = [50, 3000];
ranges.Kv_mD = [1, 300];
ranges.S = [-3, 8];
ranges.C_m3_per_MPa = [1e-10, 1e-6];
ranges.q_m3_per_d = [80, 80];
ranges.mu_cP = [5, 5];
ranges.As_in2 = [0.21, 0.21];
ranges.hw_m = [10, 10];
end

function [Dp, Derivative, status, message] = run_one_sample(sample)
try
    result = S2_Singleprobe_Sampling_V14_compute_curve(sample, 0, false, []);
    Dp = result.Dp;
    Derivative = result.derivative;
    if all(isfinite(Dp)) && all(isfinite(Derivative))
        status = "ok";
        message = "";
    else
        status = "invalid";
        message = "non-finite output";
    end
catch ME
    Dp = nan(size(sample.t_hours));
    Derivative = nan(size(sample.t_hours));
    status = "failed";
    message = string(ME.message);
end
end

function x = log_uniform(bounds)
lo = log10(bounds(1));
hi = log10(bounds(2));
x = 10^(lo + rand() * (hi - lo));
end

function x = uniform_range(bounds)
x = bounds(1) + rand() * (bounds(2) - bounds(1));
end

function val = get_option(options, field_name, default_val)
if isstruct(options) && isfield(options, field_name) && ~isempty(options.(field_name))
    val = options.(field_name);
else
    val = default_val;
end
end

function print_progress(i, N)
if i == 1 || mod(i, max(1, floor(N / 20))) == 0 || i == N
    fprintf('  %d / %d\n', i, N);
end
end
