function X = signal_tensor(x, M)
    % x is the original signal, a vector of dimension N
    % M represents m.c.e(H), indicating the outer product needs to be calculated up to (M-1)th order
    % The returned X is an (M-1)-order N-dimensional tensor

    % Get the length of the vector
    N = length(x);
    
    % Initialize x as an N×1 column vector
    x = x(:);

    % Initialize the tensor as x
    X = x;

    % Iteratively compute the outer product
    for m = 2:M-1
        % Use the Kronecker product to compute the outer product
        X = bsxfun(@times, X, reshape(x, [ones(1,m-1), N]));
    end
    tensor_dims = size(X);
    
    % Add a dimension of size 1 in the second dimension
    new_dims = [tensor_dims(1), 1, tensor_dims(2:end)];
    
    % Reshape the tensor to fit the new dimension
    X = reshape(X, new_dims);
end
