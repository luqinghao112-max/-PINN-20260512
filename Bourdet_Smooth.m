% --- Bourdet 导数 (边界优化版) ---
function dP = Bourdet_Smooth(t, p, L)
    n  = length(t); 
    dP = zeros(size(p));
    if n < 3, return; end 
    
    log10_t = log10(t); 
    ln_t    = log(t);      
    for i = 1:n
        d_left  = log10_t(i) - log10_t(1);
        d_right = log10_t(n) - log10_t(i);
        L_eff   = min([L, d_left, d_right]) + 1e-5; 
        
        k = i; while k > 1 && (log10_t(i) - log10_t(k)) < L_eff, k = k - 1; end
        j = i; while j < n && (log10_t(j) - log10_t(i)) < L_eff, j = j + 1; end
        
        if k == i && i > 1, k = i - 1; end
        if j == i && i < n, j = i + 1; end
        
        dx_L = ln_t(i) - ln_t(k); m_L  = 0; if dx_L ~= 0, m_L = (p(i) - p(k)) / dx_L; end
        dx_R = ln_t(j) - ln_t(i); m_R  = 0; if dx_R ~= 0, m_R = (p(j) - p(i)) / dx_R; end
        
        if dx_L == 0 && dx_R == 0, dP(i) = 0;
        elseif dx_L == 0, dP(i) = m_R;
        elseif dx_R == 0, dP(i) = m_L;
        else, dP(i) = (m_L * dx_R + m_R * dx_L) / (dx_L + dx_R);
        end
    end
end