% Generate dataset for the inhomogeneous Helmholtz equation:
%   u_xx + u_yy + k^2 * u = f,  on [0,1]^2
%   u = 0 on boundary (Dirichlet)
%
% The forcing f is sampled from a Gaussian Random Field (GRF).
% Saves f_data and psi_data to data/helmholtz_N-S-S_<round>.mat
%
% Usage:
%   generate_inhom_helmholtz(1000, 64, 1)   % 1000 samples, 64x64, k=1

function generate_inhom_helmholtz(N, S, k)

    if nargin < 1, N = 1000; end
    if nargin < 2, S = 64;   end
    if nargin < 3, k = 1;    end

    f_data   = zeros(N, S, S);
    psi_data = zeros(N, S, S);

    h = 1 / (S - 1);

    % 1D finite-difference Laplacian with Dirichlet BC
    e = ones(S, 1);
    L1d = spdiags([e -2*e e], -1:1, S, S) / h^2;
    L1d(1, :) = 0; L1d(1, 1) = 1;   % Dirichlet at x=0
    L1d(S, :) = 0; L1d(S, S) = 1;   % Dirichlet at x=1

    % 2D Laplacian via Kronecker product
    L_full = kron(speye(S), L1d) + kron(L1d, speye(S));

    % Helmholtz operator: A u = f
    A = L_full + k^2 * speye(S^2);

    % GRF parameters
    alpha = 2;
    tau   = 3;

    for i = 1:N
        f = 100 * GRF(alpha, tau, S);
        % Enforce zero boundary
        f(1,:) = 0; f(S,:) = 0; f(:,1) = 0; f(:,S) = 0;
        f_data(i,:,:) = f;

        f_vec   = reshape(f, [S^2, 1]);
        psi_vec = A \ f_vec;
        psi     = reshape(psi_vec, [S, S]);
        psi_data(i,:,:) = psi;

        if mod(i, 100) == 0
            fprintf('Generated %d / %d samples\n', i, N);
        end
    end

    % Save
    if ~exist('data', 'dir'), mkdir('data'); end
    filename = sprintf('data/helmholtz_%d_%d_k%d.mat', N, S, k);
    save(filename, 'f_data', 'psi_data');
    fprintf('Saved to %s\n', filename);

end
