function TV = t_tv(shiftOperator,tSignal)
% T_TV  Total variation of a tensor signal w.r.t. a shifting operator,
%   TV = |X - A * X|, with * the t-product and |.| as in T_NORM.
TV = t_norm(tSignal-t_product_fft(shiftOperator,tSignal));
end