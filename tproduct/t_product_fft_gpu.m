function C = t_product_fft_gpu(varargin)
% T_PRODUCT_FFT_GPU  GPU version of T_PRODUCT_FFT (needs Parallel Computing Toolbox).
%   C = t_product_fft_gpu(A, B, ...) returns A * B * ... under the t-product.
    if nargin < 2
        error('At least two tensors are required as input');
    end

    % Get the size and order of the first tensor
    A = varargin{1};
    P = length(size(A));
    tsize1 = size(A);
    num_slices = prod(tsize1(3:end));
    flatten_shape1 = [tsize1(1:2), num_slices];

    % Transfer tensors to GPU and Perform FFT on the first tensor
    A_fft = gpuArray(double(A));
    for i = 3:P
        A_fft = fft(A_fft, [], i);
    end
    flatten_A_fft = reshape(A_fft, flatten_shape1);

    % Perform FFT on the remaining tensors and multiply with corresponding layers of the first tensor
    for k = 2:nargin
        B = varargin{k};
        tsize2 = size(B);
        if length(size(B)) ~= P || size(flatten_A_fft, 2) ~= tsize2(1)
            error('The order and dimensions of tensors must match');
        end

        B_fft = gpuArray(double(B));
        for i = 3:P
            B_fft = fft(B_fft, [], i);
        end
        flatten_B_fft = reshape(B_fft, [tsize2(1:2), num_slices]);

        % Temporary variable to hold intermediate results
        temp_result = gpuArray(zeros([tsize1(1), tsize2(2), num_slices]));
        for i = 1:num_slices
            temp_result(:, :, i) = flatten_A_fft(:, :, i) * flatten_B_fft(:, :, i);
        end
        flatten_A_fft = temp_result;
    end

    % Reshape the product into tensor and perform IFFT
    tsize3 = [tsize1(1), size(varargin{end}, 2), tsize1(3:end)];
    C_fft = reshape(flatten_A_fft, tsize3);
    C = C_fft;
    % 'symmetric' is only valid for the last inverse transform (dim 3) and only when all inputs are real
    allReal = all(cellfun(@isreal, varargin));
    for i = P:-1:3
        if allReal && i == 3
            C = ifft(C, [], i, 'symmetric');
        else
            C = ifft(C, [], i);
        end
    end

    % Extract the real part from the complex result (if applicable) and gather the result from GPU
    tol = gpuArray(eps);
    powerIm = sum(imag(C(:)).^2);
    if powerIm < tol
        C = real(C);
    end
    C = gather(C); % Transfer the result back to CPU
end
