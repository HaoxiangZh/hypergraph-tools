% RUN_TESTS  Numerical checks for the t-product utilities.
%   Run from any folder:  run('tests/run_tests.m')
%   Each check reports the max absolute error; the script errors if any fail.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'tproduct'), fullfile(root,'hgsp'), ...
        fullfile(root,'visualization'), fullfile(root,'tests'));

rng(0);
tp = @t_product_fft;
T = {};

% --- 3rd order ---
A = rand(3,4,5); B = rand(4,2,5); C = rand(2,3,5);
T{end+1} = {'t_product vs t_product_fft (3rd)', @() t_product(A,B) - tp(A,B)};
T{end+1} = {'t_product_fft vs bcirc definition (3rd)', @() unfold(tp(A,B)) - bcirc(A)*unfold(B)};
T{end+1} = {'t_product_fft multi-arg (3rd)', @() tp(A,B,C) - tp(tp(A,B),C)};
T{end+1} = {'ttranspose (AB)^T = B^T A^T (3rd)', @() ttranspose(tp(A,B)) - tp(ttranspose(B),ttranspose(A))};
S = rand(4,4,5);
I3 = zeros(4,4,5); I3(:,:,1) = eye(4);
T{end+1} = {'t_inverse A*inv(A) = I (3rd)', @() tp(S,t_inverse(S)) - I3};
T{end+1} = {'t_power(A,2) = A*A (3rd)', @() t_power(S,2) - tp(S,S)};
T{end+1} = {'t_power(A,-1) = t_inverse(A) (3rd)', @() t_power(S,-1) - t_inverse(S)};
Ss = S + ttranspose(S);
[V,L] = t_eigendecomposition(Ss);
T{end+1} = {'t_eigendecomposition A*V = V*L (3rd)', @() ref_tprod(Ss,V) - ref_tprod(V,L)};
N = t_norm(B);
T{end+1} = {'t_norm |X|*|X| = X^T*X (3rd)', @() ref_tprod(N,N) - tp(ttranspose(B),B)};
x = rand(4,1,5); nx = t_norm(x);
T{end+1} = {'t_norm of a signal (N x 1 x n3)', @() ref_tprod(nx,nx) - tp(ttranspose(x),x)};
D = diag_circ(A); mask = kron(eye(5),ones(3,4)); Ah = fft(A,[],3);
T{end+1} = {'diag_circ is block diagonal (3rd)', @() D.*(1-mask)};
T{end+1} = {'diag_circ blocks = fft slices (3rd)', @() D(7:9,9:12) - Ah(:,:,3)};
T{end+1} = {'t_product_entrywise vs t_product_fft (3rd)', @() t_product_entrywise(A,B) - tp(A,B)};
T{end+1} = {'t_hadamard = ifft(fft(A).*fft(B)) (3rd)', @() t_hadamard(S,S) - ifft(fft(S,[],3).*fft(S,[],3),[],3)};

% --- higher order, against an independent reference ---
A4 = rand(3,4,3,2); B4 = rand(4,2,3,2); S4 = rand(3,3,3,2);
I4 = zeros(3,3,3,2); I4(:,:,1,1) = eye(3);
T{end+1} = {'t_product vs reference (4th)', @() t_product(A4,B4) - ref_tprod(A4,B4)};
T{end+1} = {'t_product_fft vs reference (4th)', @() tp(A4,B4) - ref_tprod(A4,B4)};
T{end+1} = {'ttranspose (AB)^T = B^T A^T (4th)', @() ttranspose(tp(A4,B4)) - tp(ttranspose(B4),ttranspose(A4))};
T{end+1} = {'t_inverse A*inv(A) = I (4th)', @() ref_tprod(S4,t_inverse(S4)) - I4};
T{end+1} = {'t_power(A,3) vs reference (4th)', @() t_power(S4,3) - ref_tprod(ref_tprod(S4,S4),S4)};
S4s = S4 + ttranspose(S4);
[V4,L4] = t_eigendecomposition(S4s);
T{end+1} = {'t_eigendecomposition A*V = V*L (4th)', @() ref_tprod(S4s,V4) - ref_tprod(V4,L4)};
Ac = A4 + 1i*rand(size(A4)); Bc = B4 + 1i*rand(size(B4));
T{end+1} = {'t_product_fft complex input (4th)', @() tp(Ac,Bc) - ref_tprod(Ac,Bc)};
A5 = rand(2,3,2,3,2); B5 = rand(3,2,2,3,2);
T{end+1} = {'t_product vs reference (5th)', @() t_product(A5,B5) - ref_tprod(A5,B5)};
T{end+1} = {'t_product_fft vs reference (5th)', @() tp(A5,B5) - ref_tprod(A5,B5)};

% --- hgsp helpers ---
T{end+1} = {'signal_tensor size (N=4, M=4)', @() double(~isequal(size(signal_tensor(1:4,4)),[4 1 4 4]))};
T{end+1} = {'t_sym size (3x3x3 -> 3x3x7)', @() double(~isequal(size(t_sym(rand(3,3,3))),[3 3 7]))};
T{end+1} = {'t_sym size (3x3x3x3 -> 3x3x7x7)', @() double(~isequal(size(t_sym(rand(3,3,3,3))),[3 3 7 7]))};
T{end+1} = {'random_tensor size', @() double(~isequal(size(random_tensor([3 3 3])),[3 3 7]))};
T{end+1} = {'t_tv_laplacian size', @() double(~isequal(size(t_tv_laplacian(Ss, rand(4,1,5))),[1 1 5]))};

% --- GPU (skipped when unavailable) ---
try
    hasGpu = gpuDeviceCount > 0;
catch
    hasGpu = false;
end
if hasGpu
    T{end+1} = {'t_product_fft_gpu vs reference (4th)', @() t_product_fft_gpu(A4,B4) - ref_tprod(A4,B4)};
end

nFail = 0;
for k = 1:numel(T)
    name = T{k}{1};
    try
        e = T{k}{2}();
        err = max(abs(e(:)));
        pass = err < 1e-8;
        if pass, status = 'PASS'; else, status = 'FAIL'; end
        fprintf('%-45s %s  err=%.2e\n', name, status, err);
    catch ME
        pass = false;
        fprintf('%-45s ERROR  %s\n', name, ME.message);
    end
    nFail = nFail + ~pass;
end
fprintf('\n%d/%d checks passed\n', numel(T) - nFail, numel(T));
if nFail > 0
    error('run_tests: %d check(s) failed', nFail);
end
