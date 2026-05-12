function summary = Validate_Inversion_With_V14_4Param(csv_file, mat_file, out_dir, max_rows)
% Validate_Inversion_With_V14_4Param
% 将 Python 4 参数反演结果代回 MATLAB V14 正演模型复算。
%
% 目的：
%   1) 读取反演结果 CSV 中的 est_Kh、est_Kv、est_S、est_C；
%   2) 固定已知参数 q、mu、As、hw；
%   3) 调用 S2_Singleprobe_Sampling_V14_compute_curve 重新计算压力和 Bourdet 导数；
%   4) 与原始合成数据中的压力和导数对比；
%   5) 输出 V14 复算对比图和误差统计表。
%
% 默认调用：
%   Validate_Inversion_With_V14_4Param
%
% 输出：
%   pinn_homogeneous_4param/v14_validation_outputs/v14_validation_summary.csv
%   pinn_homogeneous_4param/v14_validation_figures/v14_recheck_*.png
%   pinn_homogeneous_4param/v14_validation_figures/quality_*.png

if nargin < 1 || isempty(csv_file)
    csv_file = fullfile('pinn_homogeneous_4param', 'inversion_outputs', ...
        'inversion_results_homogeneous_4param.csv');
end
if nargin < 2 || isempty(mat_file)
    mat_file = 'singleprobe_synthetic_homogeneous_4param_V14_2000.mat';
end
if nargin < 3 || isempty(out_dir)
    out_dir = fullfile('pinn_homogeneous_4param', 'v14_validation_outputs');
end
if nargin < 4 || isempty(max_rows)
    max_rows = inf;
end

fig_dir = fullfile('pinn_homogeneous_4param', 'v14_validation_figures');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

fprintf('Reading inversion CSV: %s\n', csv_file);
T = readtable(csv_file, 'VariableNamingRule', 'preserve');

fprintf('Reading synthetic MAT: %s\n', mat_file);
Sdata = load(mat_file, 'X', 'Y', 'D', 't_hours');
X = Sdata.X;
Y = Sdata.Y;
D = Sdata.D;
t_hours = Sdata.t_hours(:).';

n_rows = min(height(T), max_rows);
fprintf('Validating %d inversion rows with V14 forward model...\n', n_rows);

rows = repmat(make_empty_summary_row(), n_rows, 1);
pressure_rel_map = nan(n_rows, numel(t_hours));
derivative_rel_map = nan(n_rows, numel(t_hours));
derivative_logerr_map = nan(n_rows, numel(t_hours));

for i = 1:n_rows
    sample_idx_py = T.sample_idx(i);
    sample_row = sample_idx_py + 1;  % Python 为 0 基下标，MATLAB 为 1 基下标。

    observed_pressure = Y(sample_row, :);
    observed_derivative = D(sample_row, :);

    p = make_base_template(t_hours);
    p.Kh = T.est_Kh_mD(i);
    p.Kv = T.est_Kv_mD(i);
    p.S = T.est_S(i);
    p.C = T.est_C_m3_per_MPa(i);
    p.q_val_default = T.known_q_m3_per_d(i);
    p.mu = T.known_mu_cP(i);
    p.As_in2 = T.known_As_in2(i);
    p.hw = T.known_hw_m(i);

    try
        result = S2_Singleprobe_Sampling_V14_compute_curve(p, 0, false, []);
        fitted_pressure_v14 = result.Dp(:).';
        fitted_derivative_v14 = result.derivative(:).';
        status = "ok";
        message = "";
    catch ME
        fitted_pressure_v14 = nan(size(t_hours));
        fitted_derivative_v14 = nan(size(t_hours));
        status = "failed";
        message = string(ME.message);
    end

    pressure_rel = relative_error_curve(fitted_pressure_v14, observed_pressure);
    derivative_rel = relative_error_curve(fitted_derivative_v14, observed_derivative);
    derivative_log_err = log10_error_curve(fitted_derivative_v14, observed_derivative);
    pressure_rel_map(i, :) = pressure_rel;
    derivative_rel_map(i, :) = derivative_rel;
    derivative_logerr_map(i, :) = derivative_log_err;

    rows(i).sample_idx = sample_idx_py;
    rows(i).status = status;
    rows(i).message = message;
    rows(i).true_Kh_mD = T.true_Kh_mD(i);
    rows(i).est_Kh_mD = T.est_Kh_mD(i);
    rows(i).true_Kv_mD = T.true_Kv_mD(i);
    rows(i).est_Kv_mD = T.est_Kv_mD(i);
    rows(i).true_S = T.true_S(i);
    rows(i).est_S = T.est_S(i);
    rows(i).true_C_m3_per_MPa = T.true_C_m3_per_MPa(i);
    rows(i).est_C_m3_per_MPa = T.est_C_m3_per_MPa(i);
    rows(i).v14_pressure_mean_relerr = nanmean(pressure_rel);
    rows(i).v14_pressure_p95_relerr = prctile(pressure_rel(isfinite(pressure_rel)), 95);
    rows(i).v14_derivative_mean_relerr = nanmean(derivative_rel);
    rows(i).v14_derivative_median_relerr = nanmedian(derivative_rel);
    rows(i).v14_derivative_p95_relerr = prctile(derivative_rel(isfinite(derivative_rel)), 95);
    rows(i).v14_derivative_median_log10err = nanmedian(derivative_log_err);

    save_validation_plot( ...
        fullfile(fig_dir, sprintf('v14_recheck_%02d_idx_%d.png', i, sample_idx_py)), ...
        t_hours, observed_pressure, fitted_pressure_v14, ...
        observed_derivative, fitted_derivative_v14, sample_idx_py, rows(i));

    fprintf('  %d / %d, idx=%d, pressure mean rel=%.3f%%, derivative median log10err=%.3f\n', ...
        i, n_rows, sample_idx_py, 100 * rows(i).v14_pressure_mean_relerr, ...
        rows(i).v14_derivative_median_log10err);
end

summary = struct2table(rows);
summary_file = fullfile(out_dir, 'v14_validation_summary.csv');
writetable(summary, summary_file);
save_quality_plots(summary, t_hours, pressure_rel_map, derivative_rel_map, ...
    derivative_logerr_map, fig_dir);

fprintf('\nSaved:\n');
fprintf('  %s\n', summary_file);
fprintf('  %s\n', fig_dir);
fprintf('  %s\n', fullfile(fig_dir, 'quality_error_overview.png'));
fprintf('  %s\n', fullfile(fig_dir, 'quality_parameter_crossplot.png'));
fprintf('  %s\n', fullfile(fig_dir, 'quality_error_heatmap.png'));
fprintf('\nV14 validation summary:\n');
fprintf('  pressure mean relerr: mean=%.3f%%, median=%.3f%%\n', ...
    100 * nanmean(summary.v14_pressure_mean_relerr), ...
    100 * nanmedian(summary.v14_pressure_mean_relerr));
fprintf('  derivative median log10err: mean=%.3f, median=%.3f\n', ...
    nanmean(summary.v14_derivative_median_log10err), ...
    nanmedian(summary.v14_derivative_median_log10err));

end

function row = make_empty_summary_row()
row = struct();
row.sample_idx = nan;
row.status = "";
row.message = "";
row.true_Kh_mD = nan;
row.est_Kh_mD = nan;
row.true_Kv_mD = nan;
row.est_Kv_mD = nan;
row.true_S = nan;
row.est_S = nan;
row.true_C_m3_per_MPa = nan;
row.est_C_m3_per_MPa = nan;
row.v14_pressure_mean_relerr = nan;
row.v14_pressure_p95_relerr = nan;
row.v14_derivative_mean_relerr = nan;
row.v14_derivative_median_relerr = nan;
row.v14_derivative_p95_relerr = nan;
row.v14_derivative_median_log10err = nan;
end

function base = make_base_template(t_hours)
% 与 4 参数数据生成脚本保持一致的 V14 参数包。
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

function rel = relative_error_curve(pred, truth)
mask = isfinite(pred) & isfinite(truth) & truth > 0 & pred > 0;
rel = nan(size(truth));
rel(mask) = abs(pred(mask) - truth(mask)) ./ max(abs(truth(mask)), 1e-12);
end

function err = log10_error_curve(pred, truth)
mask = isfinite(pred) & isfinite(truth) & truth > 0 & pred > 0;
err = nan(size(truth));
err(mask) = abs(log10(pred(mask)) - log10(truth(mask)));
end

function save_validation_plot(out_path, t_hours, p_obs, p_fit, d_obs, d_fit, sample_idx, row)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1100, 760]);
loglog(t_hours, positive_or_nan(p_obs), 'k-', 'LineWidth', 1.8); hold on;
loglog(t_hours, positive_or_nan(p_fit), 'r--', 'LineWidth', 1.6);
loglog(t_hours, positive_or_nan(d_obs), '-.', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.4);
loglog(t_hours, positive_or_nan(d_fit), 'm:', 'LineWidth', 1.8);
grid on; grid minor;
xlabel('Time t [h]', 'FontSize', 12);
ylabel('Pressure drop and derivative [MPa]', 'FontSize', 12);
title(sprintf('V14 recheck sample %d | pressure mean rel err=%.3f%%', ...
    sample_idx, 100 * row.v14_pressure_mean_relerr), 'FontSize', 14);
legend({'Observed pressure', 'V14 pressure from inverted params', ...
    'Observed derivative', 'V14 derivative from inverted params'}, ...
    'Location', 'best');
set(gca, 'FontSize', 11);
saveas(fig, out_path);
close(fig);
end

function save_quality_plots(summary, t_hours, pressure_rel_map, derivative_rel_map, derivative_logerr_map, fig_dir)
% 生成用于论文/汇报的拟合质量汇总图。
save_error_overview_plot(summary, fullfile(fig_dir, 'quality_error_overview.png'));
save_parameter_crossplot(summary, fullfile(fig_dir, 'quality_parameter_crossplot.png'));
save_error_heatmap_plot(t_hours, summary.sample_idx, pressure_rel_map, derivative_rel_map, ...
    derivative_logerr_map, fullfile(fig_dir, 'quality_error_heatmap.png'));
end

function save_error_overview_plot(summary, out_path)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1280, 820]);
x = 1:height(summary);
sample_labels = string(summary.sample_idx);

subplot(2, 2, 1);
bar(x, 100 * summary.v14_pressure_mean_relerr, 0.75, 'FaceColor', [0.20 0.45 0.70]);
format_bar_axis(sample_labels);
ylabel('Mean relative error [%]');
title('Pressure Mean Error');
grid on; box on;

subplot(2, 2, 2);
bar(x, 100 * summary.v14_pressure_p95_relerr, 0.75, 'FaceColor', [0.20 0.65 0.50]);
format_bar_axis(sample_labels);
ylabel('P95 relative error [%]');
title('Pressure P95 Error');
grid on; box on;

subplot(2, 2, 3);
bar(x, summary.v14_derivative_median_log10err, 0.75, 'FaceColor', [0.70 0.35 0.65]);
format_bar_axis(sample_labels);
ylabel('Median |log_{10}(fit/obs)|');
title('Derivative Median Log Error');
grid on; box on;

subplot(2, 2, 4);
bar(x, 100 * summary.v14_derivative_p95_relerr, 0.75, 'FaceColor', [0.85 0.45 0.20]);
format_bar_axis(sample_labels);
ylabel('P95 relative error [%]');
title('Derivative P95 Error');
grid on; box on;

sgtitle('V14 Recalculation Fit Quality', 'FontWeight', 'bold', 'FontSize', 16);
save_png(fig, out_path);
close(fig);
end

function save_parameter_crossplot(summary, out_path)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1180, 900]);

subplot(2, 2, 1);
plot_one_to_one_log(summary.true_Kh_mD, summary.est_Kh_mD, 'K_h [mD]');
title(sprintf('K_h | median err = %.2f%%', 100 * nanmedian(abs(summary.est_Kh_mD - summary.true_Kh_mD) ./ summary.true_Kh_mD)));

subplot(2, 2, 2);
plot_one_to_one_log(summary.true_Kv_mD, summary.est_Kv_mD, 'K_v [mD]');
title(sprintf('K_v | median err = %.2f%%', 100 * nanmedian(abs(summary.est_Kv_mD - summary.true_Kv_mD) ./ summary.true_Kv_mD)));

subplot(2, 2, 3);
plot_one_to_one_linear(summary.true_S, summary.est_S, 'S');
title(sprintf('S | median abs err = %.3f', nanmedian(abs(summary.est_S - summary.true_S))));

subplot(2, 2, 4);
plot_one_to_one_log(summary.true_C_m3_per_MPa, summary.est_C_m3_per_MPa, 'C [m^3/MPa]');
title(sprintf('C | median log err = %.3f', ...
    nanmedian(abs(log10(summary.est_C_m3_per_MPa) - log10(summary.true_C_m3_per_MPa)))));

sgtitle('True vs Inverted Parameters', 'FontWeight', 'bold', 'FontSize', 16);
save_png(fig, out_path);
close(fig);
end

function save_error_heatmap_plot(t_hours, sample_idx, pressure_rel_map, derivative_rel_map, derivative_logerr_map, out_path)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1280, 900]);
x = log10(t_hours);
y = 1:numel(sample_idx);

subplot(3, 1, 1);
imagesc(x, y, log10(max(pressure_rel_map, 1e-6)));
set(gca, 'YDir', 'normal');
colorbar; caxis([-5, 0]);
ylabel('Sample');
title('Pressure log_{10}(relative error)');
format_heatmap_axis(sample_idx);

subplot(3, 1, 2);
imagesc(x, y, log10(max(derivative_rel_map, 1e-6)));
set(gca, 'YDir', 'normal');
colorbar; caxis([-5, 1]);
ylabel('Sample');
title('Derivative log_{10}(relative error)');
format_heatmap_axis(sample_idx);

subplot(3, 1, 3);
imagesc(x, y, derivative_logerr_map);
set(gca, 'YDir', 'normal');
colorbar; caxis([0, max(0.15, min(0.5, nanpercentile_local(derivative_logerr_map(:), 95)))]);
ylabel('Sample');
xlabel('log_{10}(time [h])');
title('Derivative |log_{10}(fit/obs)|');
format_heatmap_axis(sample_idx);

sgtitle('Time-Resolved V14 Recalculation Errors', 'FontWeight', 'bold', 'FontSize', 16);
save_png(fig, out_path);
close(fig);
end

function format_bar_axis(sample_labels)
set(gca, 'XTick', 1:numel(sample_labels), 'XTickLabel', sample_labels, ...
    'FontSize', 10, 'LineWidth', 1.0);
xlabel('Sample index');
try
    xtickangle(45);
catch
end
end

function format_heatmap_axis(sample_idx)
set(gca, 'YTick', 1:numel(sample_idx), 'YTickLabel', string(sample_idx), ...
    'FontSize', 10, 'LineWidth', 1.0);
end

function plot_one_to_one_log(true_val, est_val, axis_label)
mask = isfinite(true_val) & isfinite(est_val) & true_val > 0 & est_val > 0;
loglog(true_val(mask), est_val(mask), 'o', 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.20 0.45 0.70], 'MarkerEdgeColor', 'k'); hold on;
lims = axis_limits_positive([true_val(mask); est_val(mask)]);
loglog(lims, lims, 'k--', 'LineWidth', 1.2);
xlim(lims); ylim(lims);
xlabel(['True ', axis_label]);
ylabel(['Inverted ', axis_label]);
grid on; box on;
set(gca, 'FontSize', 11, 'LineWidth', 1.0);
end

function plot_one_to_one_linear(true_val, est_val, axis_label)
mask = isfinite(true_val) & isfinite(est_val);
plot(true_val(mask), est_val(mask), 'o', 'MarkerSize', 6, ...
    'MarkerFaceColor', [0.20 0.45 0.70], 'MarkerEdgeColor', 'k'); hold on;
lims = axis_limits_linear([true_val(mask); est_val(mask)]);
plot(lims, lims, 'k--', 'LineWidth', 1.2);
xlim(lims); ylim(lims);
xlabel(['True ', axis_label]);
ylabel(['Inverted ', axis_label]);
grid on; box on;
set(gca, 'FontSize', 11, 'LineWidth', 1.0);
end

function lims = axis_limits_positive(v)
v = v(isfinite(v) & v > 0);
if isempty(v)
    lims = [1, 10];
    return;
end
lo = 10 ^ floor(log10(min(v)) - 0.1);
hi = 10 ^ ceil(log10(max(v)) + 0.1);
lims = [lo, hi];
end

function lims = axis_limits_linear(v)
v = v(isfinite(v));
if isempty(v)
    lims = [-1, 1];
    return;
end
lo = min(v);
hi = max(v);
pad = 0.08 * max(hi - lo, 1);
lims = [lo - pad, hi + pad];
end

function q = nanpercentile_local(v, pct)
v = v(isfinite(v));
if isempty(v)
    q = nan;
else
    q = prctile(v, pct);
end
end

function save_png(fig, out_path)
try
    exportgraphics(fig, out_path, 'Resolution', 300);
catch
    print(fig, out_path, '-dpng', '-r300');
end
end

function y = positive_or_nan(x)
y = x;
y(~isfinite(y) | y <= 0) = nan;
end
