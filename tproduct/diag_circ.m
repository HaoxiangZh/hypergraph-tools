function A_hat = diag_circ(A)
% DIAG_CIRC  Block-diagonalize bcirc(A) with the DFT matrix.
%   For a 3rd-order tensor, (F kron I) * bcirc(A) * (F kron I)' / n3 is
%   block diagonal with blocks equal to the frontal slices of fft(A,[],3).
if(length(size(A))==2)
    F = fft(eye(length(A)));
    A_circ = bcirc(A);
    F_ct = F';
    A_hat = F*A_circ*F_ct;
elseif(length(size(A))==3)
    F = kron(fft(eye(size(A,3))),eye(size(A,1)));
    A_circ = bcirc(A);
    F_ct = kron(fft(eye(size(A,3))),eye(size(A,2)));
    A_hat = F*A_circ*F_ct'/size(A,3);
end
return