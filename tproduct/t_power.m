function C = t_power(A, power)
    % t_power computes the "power" of a tensor along its frontal slices.
    % A is a multi-dimensional array (tensor), and "power" is the exponent.
    % The function applies the power operation to each frontal slice of the tensor in the Fourier domain.
    % The result is stored in C.

    % Get the size of the tensor A
    tsize = size(A);
    % P is the number of dimensions(order) in the tensor A
    P = length(tsize);
    % Calculate the number of frontal slices by multiplying the sizes of the 3rd to P-th dimensions
    num_slices = prod(tsize(3:P));

    % Initialize A_hat as a copy of A to store its Fourier transform
    A_hat = A;
    % Initialize C_hat and C to store the results of the computations
    C_hat = zeros(tsize);
    C = zeros(tsize);

    % Apply the FFT (Fast Fourier Transform) along each dimension from 3 to P
    for i = 3:P
        A_hat = fft(A_hat, [], i);
    end

    % For each frontal slice, compute the matrix raised to the given power
    for i = 1:num_slices
        C_hat(:,:,i) = A_hat(:,:,i)^power;
    end

    % Copy the result into C
    C = C_hat;

    % Apply the inverse FFT along each dimension from 3 to P
    for i = 3:P
        C = ifft(C, [], i);
    end 
end
