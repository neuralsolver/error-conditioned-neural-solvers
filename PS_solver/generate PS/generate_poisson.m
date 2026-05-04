% function generate_poisson(N, S)
%     % Default arguments
%     for round = 1:2
%         if nargin < 1
%             N = 100; % Number of generations
%         end
%         if nargin < 2
%             S = 256; % Resolution
%         end
% 
%         % Preallocate arrays
%         f_data = zeros(N, S, S);
%         phi_data = zeros(N, S, S);
% 
%         % Parameters for GRF
%         alpha = 2;
%         tau = 3;
% 
%         for i = 1:N
%             % Generate the coefficient f using GRF
%             f = 100 * GRF(alpha, tau, S);
% 
%             % Solve the Poisson equation for phi
%             phi = solve_poisson(f, S);
% 
%             % Store the generated data
%             f_data(i, :, :) = f;
%             phi_data(i, :, :) = phi;
%         end
% 
%         % Save the dataset
%         if ~exist('data', 'dir')
%             mkdir('data');
%         end
%         filename = sprintf('data/poisson_%d-%d-%d_%d.mat', N, S, S, round);
%         save(filename, 'f_data', 'phi_data');
%     end
% end
% 
% function phi = solve_poisson(f, S)
%     % Assuming f is already on a SxS grid and represents the source term
%     % uniformly distributed across the domain [0,1]x[0,1].
% 
%     % Define grid spacing
%     h = 1 / (S - 1);
% 
%     % Initialize phi
%     phi = zeros(S, S);
% 
%     % Assemble the system matrix for the Poisson equation
%     % For simplicity, using a 5-point Laplacian stencil with Dirichlet boundary conditions
%     N = S^2; % Total number of points in the grid
%     A = sparse(N, N);
%     B = 2 * reshape(f, [N, 1]); % Reshape f into a vector for the linear system
% 
%     for i = 1:S
%         for j = 1:S
%             index = (i - 1) * S + j; % Convert (i, j) to linear index
% 
%             % Apply Dirichlet boundary conditions: phi = 0 at boundaries
%             if i == 1 || i == S || j == 1 || j == S
%                 A(index, index) = 1; % Boundary points
%                 B(index) = 0; % Assuming phi = 0 on the boundary
%             else
%                 % Internal points - Discretize Laplacian operator
%                 A(index, index) = -4 / h^2;
%                 A(index, index - 1) = 1 / h^2; % Left
%                 A(index, index + 1) = 1 / h^2; % Right
%                 A(index, index - S) = 1 / h^2; % Below
%                 A(index, index + S) = 1 / h^2; % Above
%             end
%         end
%     end
% 
%     % Solve the linear system A*phi = B
%     phi_vec = A \ B;
% 
%     % Reshape the solution back to a 2D grid
%     phi = reshape(phi_vec, [S, S]);
% end


function generate_poisson(N, S)
    if nargin < 1
        N = 100;
    end
    if nargin < 2
        S = 128;
    end

    if ~exist('data', 'dir')
        mkdir('data');
    end

    alpha = 2;
    tau = 3;

    [A, h] = build_poisson_matrix(S);

    for round = 1:2
        f_data = zeros(N, S, S);
        phi_data = zeros(N, S, S);

        for i = 1:N
            f = 100 * GRF(alpha, tau, S);
            phi = solve_poisson_fast(f, A, S);

            f_data(i, :, :) = f;
            phi_data(i, :, :) = phi;
        end

        filename = sprintf('data/poisson_%d-%d-%d_%d.mat', N, S, S, round);
        save(filename, 'f_data', 'phi_data');
    end
end

function [A, h] = build_poisson_matrix(S)
    h = 1 / (S - 1);
    N = S^2;
    A = sparse(N, N);

    for i = 1:S
        for j = 1:S
            index = (i - 1) * S + j;

            if i == 1 || i == S || j == 1 || j == S
                A(index, index) = 1;
            else
                A(index, index) = -4 / h^2;
                A(index, index - 1) = 1 / h^2;
                A(index, index + 1) = 1 / h^2;
                A(index, index - S) = 1 / h^2;
                A(index, index + S) = 1 / h^2;
            end
        end
    end
end

function phi = solve_poisson_fast(f, A, S)
    N = S^2;
    B = reshape(f, [N, 1]);

    % zero Dirichlet BC
    for i = 1:S
        for j = 1:S
            if i == 1 || i == S || j == 1 || j == S
                index = (i - 1) * S + j;
                B(index) = 0;
            end
        end
    end

    phi_vec = A \ B;
    phi = reshape(phi_vec, [S, S]);
end