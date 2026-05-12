% 6_check_matlab
% 将第 5 步网络反演得到的 Kh/Kv/S/C 代回 MATLAB 正演复算。
%
% 运行方式：
%   直接点击 MATLAB 编辑器的 Run，或在命令行执行：
%   run('6_check_matlab.m')
%
% 默认输入：
%   nn_flow/out/fit_998.json
%   data_2000.mat
%
% 默认输出：
%   nn_flow/mat_out/summary.csv
%   nn_flow/mat_fig/check_998.png

result_json = fullfile('nn_flow', 'out', 'fit_998.json');
mat_file = 'data_2000.mat';
out_dir = fullfile('nn_flow', 'mat_out');

summary = check_matlab_impl(result_json, mat_file, out_dir);

function summary = check_matlab_impl(result_json, mat_file, out_dir)

if nargin < 1 || isempty(result_json)
    result_json = fullfile('nn_flow', 'out', 'fit_998.json');
end
if nargin < 2 || isempty(mat_file)
    mat_file = 'data_2000.mat';
end
if nargin < 3 || isempty(out_dir)
    out_dir = fullfile('nn_flow', 'mat_out');
end

fig_dir = fullfile('nn_flow', 'mat_fig');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

fprintf('Reading inversion result JSON: %s\n', result_json);
R = jsondecode(fileread(result_json));
sample_idx_py = R.sample_idx;
sample_row = sample_idx_py + 1;

fprintf('Reading synthetic MAT: %s\n', mat_file);
Sdata = load(mat_file, 'X', 'Y', 'D', 't_hours');
X = Sdata.X;
Y = Sdata.Y;
D = Sdata.D;
t_hours = Sdata.t_hours(:).';

observed_pressure = Y(sample_row, :);
observed_derivative = D(sample_row, :);

p = make_base_template_full_pinn(t_hours);
p.Kh = R.estimated.Kh_mD;
p.Kv = R.estimated.Kv_mD;
p.S = R.estimated.S;
p.C = R.estimated.C_m3_per_MPa;
p.q_val_default = X(sample_row, 5);
p.mu = X(sample_row, 6);
p.As_in2 = X(sample_row, 7);
p.hw = X(sample_row, 8);

fprintf('Rechecking with MATLAB forward model, sample_idx=%d...\n', sample_idx_py);
result = S2_Singleprobe_Sampling_V14_compute_curve(p, 0, false, []);
fitted_pressure_v14 = result.Dp(:).';
fitted_derivative_v14 = result.derivative(:).';

pressure_rel = relative_error_curve_full_pinn(fitted_pressure_v14, observed_pressure);
derivative_rel = relative_error_curve_full_pinn(fitted_derivative_v14, observed_derivative);
derivative_log_err = log10_error_curve_full_pinn(fitted_derivative_v14, observed_derivative);

summary = table();
summary.sample_idx = sample_idx_py;
summary.true_Kh_mD = R.true.Kh_mD;
summary.est_Kh_mD = R.estimated.Kh_mD;
summary.true_Kv_mD = R.true.Kv_mD;
summary.est_Kv_mD = R.estimated.Kv_mD;
summary.true_S = R.true.S;
summary.est_S = R.estimated.S;
summary.true_C_m3_per_MPa = R.true.C_m3_per_MPa;
summary.est_C_m3_per_MPa = R.estimated.C_m3_per_MPa;
summary.phys_pressure_mean_relerr = nanmean(pressure_rel);
summary.phys_pressure_p95_relerr = prctile(pressure_rel(isfinite(pressure_rel)), 95);
summary.phys_derivative_mean_relerr = nanmean(derivative_rel);
summary.phys_derivative_median_relerr = nanmedian(derivative_rel);
summary.phys_derivative_p95_relerr = prctile(derivative_rel(isfinite(derivative_rel)), 95);
summary.phys_derivative_median_log10err = nanmedian(derivative_log_err);

summary_file = fullfile(out_dir, 'summary.csv');
writetable(summary, summary_file);

fig_path = fullfile(fig_dir, sprintf('check_%d.png', sample_idx_py));
save_validation_plot_full_pinn(fig_path, t_hours, observed_pressure, fitted_pressure_v14, ...
    observed_derivative, fitted_derivative_v14, sample_idx_py, summary);

fprintf('\nSaved:\n');
fprintf('  %s\n', summary_file);
fprintf('  %s\n', fig_path);
fprintf('\nNetwork -> MATLAB validation:\n');
fprintf('  pressure mean relerr = %.3f%%\n', 100 * summary.phys_pressure_mean_relerr);
fprintf('  derivative median log10err = %.3f\n', summary.phys_derivative_median_log10err);

end

function base = make_base_template_full_pinn(t_hours)
% 与第 1 步数据生成脚本保持一致的参数包。
base = struct();
base.model_choice = 1;
base.param_choice = 0;
base.t_hours = t_hours;

base.phi = 0.2;
base.ct = 5e-4;
base.phi_ct = base.phi * base.ct;
base.h = 20;
base.probe_shape = 'circle';
base.negative_skin_max_factor = 20;

base.omega = 1e-2;
base.lambda = 1e-8;
base.omega2 = 1e-2;
base.lambda2 = 1e-8;
base.Kh2 = 1;
base.Kv2 = 0.5;
base.Ri = 1000;

base.Np_probe_target = 5;
base.stehfest_N = 10;
base.series_min_terms = 10;
base.series_max_terms = 80;
base.series_tol = 1e-6;
base.series_small_count = 3;

base.Kh = 1000;
base.Kv = 50;
base.S = 1;
base.C = 1e-7;
base.q_val_default = 80;
base.mu = 5;
base.As_in2 = 0.21;
base.hw = 10;
end

function rel = relative_error_curve_full_pinn(pred, truth)
mask = isfinite(pred) & isfinite(truth) & truth > 0 & pred > 0;
rel = nan(size(truth));
rel(mask) = abs(pred(mask) - truth(mask)) ./ max(abs(truth(mask)), 1e-12);
end

function err = log10_error_curve_full_pinn(pred, truth)
mask = isfinite(pred) & isfinite(truth) & truth > 0 & pred > 0;
err = nan(size(truth));
err(mask) = abs(log10(pred(mask)) - log10(truth(mask)));
end

function save_validation_plot_full_pinn(out_path, t_hours, p_obs, p_fit, d_obs, d_fit, sample_idx, row)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1100, 760]);
loglog(t_hours, positive_or_nan_full_pinn(p_obs), 'k-', 'LineWidth', 1.8); hold on;
loglog(t_hours, positive_or_nan_full_pinn(p_fit), 'r--', 'LineWidth', 1.6);
loglog(t_hours, positive_or_nan_full_pinn(d_obs), '-.', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4);
loglog(t_hours, positive_or_nan_full_pinn(d_fit), 'm:', 'LineWidth', 1.8);
grid on; box on;
xlabel('Time t [h]');
ylabel('Pressure drop and derivative [MPa]');
title(sprintf('Network parameters checked by MATLAB | sample %d | pressure mean rel=%.3f%%', ...
    sample_idx, 100 * row.phys_pressure_mean_relerr));
legend({'Original pressure', 'MATLAB using network params', ...
    'Original derivative', 'Derivative using network params'}, ...
    'Location', 'southwest');

txt = sprintf(['True / network params\n', ...
    'Kh: %.4g / %.4g mD\n', ...
    'Kv: %.4g / %.4g mD\n', ...
    'S : %.4g / %.4g\n', ...
    'C : %.3g / %.3g m^3/MPa'], ...
    row.true_Kh_mD, row.est_Kh_mD, ...
    row.true_Kv_mD, row.est_Kv_mD, ...
    row.true_S, row.est_S, ...
    row.true_C_m3_per_MPa, row.est_C_m3_per_MPa);
annotation(fig, 'textbox', [0.58 0.17 0.32 0.22], 'String', txt, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', [0.2 0.2 0.2], ...
    'FontName', 'Consolas', 'FontSize', 10);

exportgraphics(fig, out_path, 'Resolution', 220);
close(fig);
end

function y = positive_or_nan_full_pinn(x)
y = x;
y(~isfinite(y) | y <= 0) = nan;
end
