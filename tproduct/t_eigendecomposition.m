function [V, lambda] = t_eigendecomposition(A)
    % This function computes the eigendecomposition of a 3rd or higher-order tensor.
    % The input is a 3/p-order tensor A.
    % The outputs are the eigenvalue tensor (lambda) and the eigenvector tensor (V).
    % If a slice is not diagonalizable, the Jordan decomposition is used instead of eigendecomposition.

    % Get the size of the input tensor A
    tsize = size(A);
    
    % Compute the number of frontal slices in the tensor (3rd to p-th dimensions)
    num_slices = prod(tsize(3:end));
    
    % M is the number of dimensions of the tensor
    M = length(tsize);
    
    % N is the size of the first dimension of A (assuming A is square in the first two dimensions)
    N = tsize(1);
    
    % Initialize A_hat as a copy of A to store the Fourier transformed tensor
    A_hat = A;
    
    % Perform FFT on A along the 3rd to M-th dimensions (to work slice-wise in the FFT domain)
    for i = 3 : M
        A_hat = fft(A_hat, [], i);  % Apply FFT along dimension i
    end

    % Initialize V_hat and lambda_hat to store the eigenvectors and eigenvalues in the FFT domain
    V_hat = zeros(tsize);      % Tensor to store the eigenvector results
    lambda_hat = zeros(tsize); % Tensor to store the eigenvalue results

    % Loop over each frontal slice of the tensor in the FFT domain
    for k = 1 : num_slices
        % Perform eigendecomposition on each frontal slice in the FFT domain
        [V, D] = eig(A_hat(:,:,k));

        threshold = 1e-10;
        V(abs(V) < threshold) = 0;
        % Check if the matrix is diagonalizable by testing the rank of the eigenvector matrix V
        if rank(V) < N
            % If not diagonalizable, use Jordan decomposition instead
            [V, D] = jordan(A_hat(:,:,k));
        end
        
        % Store the eigenvectors (V) and eigenvalues (D) in V_hat and lambda_hat respectively
        V_hat(:,:,k) = V;
        lambda_hat(:,:,k) = D;
    end

    % Reshape the eigenvalue and eigenvector tensors to match the input tensor size
    lambda = lambda_hat;
    V = V_hat;

    % Perform inverse FFT to bring the results back from the frequency domain to the original domain
    for i = M :-1: 3
        lambda = ifft(lambda, [], i);  % Inverse FFT for eigenvalue tensor
        V = ifft(V, [], i);            % Inverse FFT for eigenvector tensor
    end

    % Return the eigenvector tensor V and eigenvalue tensor lambda
    return
end
