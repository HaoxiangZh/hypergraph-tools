% GENERATE_MATLAB_REFERENCE  Save inputs/outputs of the MATLAB toolbox so the
% Python port can be checked against it (tests/test_matlab_parity.py).
%   Run from the repository root:  run('python/tests/data/generate_matlab_reference.m')

here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(fullfile(root,'tproduct'), fullfile(root,'hgsp'));

rng(42);
A3 = rand(3,4,5); B3 = rand(4,2,5); C3 = rand(2,3,5); S3 = rand(4,4,5);
A4 = rand(3,4,3,2); B4 = rand(4,2,3,2); S4 = rand(3,3,3,2);
Ac4 = A4 + 1i*rand(size(A4)); Bc4 = B4 + 1i*rand(size(B4));
A5 = rand(2,3,2,3,2); B5 = rand(3,2,2,3,2);
T3 = rand(3,3,3); T4 = rand(3,3,3,3);
Ss3 = S3 + ttranspose(S3);
xs = rand(4,1,5);
x = [1 2 3 4];

out = struct();
out.t_product_3 = t_product_fft(A3,B3);
out.t_product_4 = t_product_fft(A4,B4);
out.t_product_5 = t_product_fft(A5,B5);
out.t_product_complex_4 = t_product_fft(Ac4,Bc4);
out.t_product_multi_3 = t_product_fft(A3,B3,C3);
out.t_product_def_4 = t_product(A4,B4);
out.ttranspose_3 = ttranspose(A3);
out.ttranspose_4 = ttranspose(A4);
out.ttranspose_complex_4 = ttranspose(Ac4);
out.t_inverse_3 = t_inverse(S3);
out.t_inverse_4 = t_inverse(S4);
out.t_power_3_2 = t_power(S3,2);
out.t_power_3_m1 = t_power(S3,-1);
out.t_power_4_3 = t_power(S4,3);
out.t_hadamard_3 = t_hadamard(S3,Ss3);
out.t_norm_3 = t_norm(B3);
out.bcirc_3 = bcirc(A3);
out.bcirc_4 = bcirc(A4);
out.unfold_4 = unfold(A4);
out.diag_circ_3 = diag_circ(A3);
out.t_sym_3 = t_sym(T3);
out.t_sym_4 = t_sym(T4);
out.signal_tensor_4 = signal_tensor(x,4);
out.t_tv_3 = t_tv(Ss3,xs);
out.t_tv_laplacian_3 = t_tv_laplacian(Ss3,xs);

save(fullfile(here,'matlab_reference.mat'), ...
     'A3','B3','C3','S3','A4','B4','S4','Ac4','Bc4','A5','B5','T3','T4','Ss3','xs','x','out','-v7');
fprintf('saved %s\n', fullfile(here,'matlab_reference.mat'));
