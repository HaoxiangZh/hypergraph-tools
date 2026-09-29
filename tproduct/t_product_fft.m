function C = t_product_fft(varargin)
    % This function computes the tensor product of multiple tensors using the FFT (Fast Fourier Transform).
    % The inputs are a variable number of tensors, and the result is the product of all input tensors.
    % The product is computed slice by slice after applying FFT to each tensor.
    
    % Check if at least two tensors are provided as input
    if nargin < 2
        error('At least two tensors are required as input');
    end

    % Get the size and order (number of dimensions) of the first tensor
    A = varargin{1};
    P = length(size(A));  % P is the number of dimensions
    tsize1 = size(A);  % Get the size of the first tensor
    num_slices = prod(tsize1(3:end));  % Calculate the number of frontal slices
    flatten_shape1 = [tsize1(1:2), num_slices];  % Flatten the shape of the first tensor for ease of manipulation

    % Perform FFT on the first tensor across all dimensions from 3 to P
    A_fft = double(A);
    for i = 3:P
        A_fft = fft(A_fft, [], i);  % Apply FFT along each dimension starting from the 3rd
    end
    flatten_A_fft = reshape(A_fft, flatten_shape1);  % Reshape the tensor for easy multiplication

    % Loop over the remaining tensors and compute the product
    for k = 2:nargin
        B = varargin{k};  % Get the next tensor
        tsize2 = size(B);  % Get the size of the current tensor
        % Ensure the tensors have the same order and compatible dimensions
        if ~isequal(tsize2(3:end), tsize1(3:end)) || size(flatten_A_fft, 2) ~= tsize2(1)
            error('The order and dimensions of tensors must match');
        end

        % Perform FFT on the current tensor across dimensions from 3 to P
        B_fft = double(B);
        for i = 3:P
            B_fft = fft(B_fft, [], i);
        end
        flatten_B_fft = reshape(B_fft, [tsize2(1:2), num_slices]);  % Reshape the current tensor for multiplication

        % Initialize a temporary variable to store the intermediate results
        temp_result = zeros([tsize1(1), tsize2(2), num_slices]);
        % Multiply corresponding frontal slices of the tensors
        for i = 1:num_slices
            temp_result(:, :, i) = flatten_A_fft(:, :, i) * flatten_B_fft(:, :, i);
        end
        % Update the flattened result with the intermediate product
        flatten_A_fft = temp_result;
    end

    % Reshape the result back into a tensor form
    tsize3 = [tsize1(1), size(varargin{end}, 2), tsize1(3:end)];  % Determine the size of the final result
    C_fft = reshape(flatten_A_fft, tsize3);  % Reshape the flattened result
    C = C_fft;
    
    % Perform inverse FFT to convert back to the original domain
    % 'symmetric' is only valid for the last inverse transform (dim 3) and only when all inputs are real
    allReal = all(cellfun(@isreal, varargin));
    for i = P:-1:3
        if allReal && i == 3
            C = ifft(C, [], i, 'symmetric');
        else
            C = ifft(C, [], i);
        end
    end

    % Check if the result is purely real and remove any small imaginary parts
    tol = eps;  % Set a tolerance for numerical errors
    powerIm = sum(imag(C(:)).^2);  % Calculate the power of the imaginary part
    if powerIm < tol  % If the imaginary part is small, remove it
        C = real(C);
    end
end
