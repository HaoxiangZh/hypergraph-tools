function C = t_hadamard(A, B)
    % t_hadamard performs the Hadamard product (element-wise multiplication) 
    % of two tensors A and B in the Fourier domain (along the third and higher dimensions).
    % A and B must have the same dimensions.

    % Get the size of tensors A and B
    tsizeA = size(A);
    tsizeB = size(B);
    
    % Check if the dimensions of A and B match
    if tsizeA ~= tsizeB
        error("dimensions do not match!");  % If not, raise an error
    end
    
    % Get the number of dimensions (order) of the tensor
    P = length(tsizeA);
    % Compute the number of slices along the third to P-th dimensions
    num_slices = prod(tsizeA(3:P));
    
    % Initialize A_hat and B_hat as copies of A and B for Fourier transformation
    A_hat = A;
    B_hat = B;
    % Initialize C_hat to store the product in the Fourier domain
    C_hat = zeros(tsizeA);
    % Initialize C to store the final result
    C = zeros(tsizeA);

    % Perform FFT on both tensors A and B along the 3rd to P-th dimensions
    for i = 3:P
        A_hat = fft(A_hat, [], i);
        B_hat = fft(B_hat, [], i);
    end

    % Perform element-wise multiplication (Hadamard product) slice by slice
    for i = 1:num_slices
        C_hat(:,:,i) = A_hat(:,:,i) .* B_hat(:,:,i);  % Element-wise multiply each frontal slice
    end

    % Copy the result of Hadamard product in the frequency domain to C
    C = C_hat;
    
    % Perform inverse FFT on the result along the 3rd to P-th dimensions
    for i = 3:P
        C = ifft(C, [], i);  % Apply inverse FFT along dimension i
    end 
end
