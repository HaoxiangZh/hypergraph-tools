function tv = t_tv_laplacian(t_Laplacian,t_signal)
% T_TV_LAPLACIAN  Laplacian quadratic form X^T * L * X under the t-product.
tv = t_product_fft(ttranspose(t_signal),t_Laplacian,t_signal);
end