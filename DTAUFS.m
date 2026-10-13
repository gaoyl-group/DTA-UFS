function [feaIdx, F, W, B, E, S, obj] = DTAUFS(X, param)
% X: dim * n
% min alpha*norm(X'*W-E, 'fro')^2 + norm(F'-B*E', 'fro')^2
%  + gamma*trace(F'*Ls*F) + eta*norm(W, 2,1) + delta*norm(Z-E, 'fro')^2
% param.solver: 'feature' (default) or 'sample'.

    c = param.c;
    d = param.d;
    pa = param.pa;
    pn = param.pn;
    pr = param.pr;
    pd = 1e4;
    NITER = 30;

    solver = 'feature';
    if isfield(param, 'solver')
        solver = param.solver;
    end
    switch solver
        case 'feature'
            useSampleSpace = false;
        case 'sample'
            useSampleSpace = true;
        otherwise
            error('DTAUFS:InvalidSolver', ...
                  'param.solver must be ''feature'' or ''sample''.');
    end
    
    [dim, num] = size(X);

    %% Construct the fixed graph S and precompute its system
    k = 5;
    [A, ~] = selftuning(X', k);
    S = (A + A') / 2;
    D1 = diag(sum(S));
    Ls = D1 - S;
    II = eye(num, num);
    F_system = II + pr * Ls;
    epsilon = 1e-8;
    if useSampleSpace
        F_system = decomposition(F_system);
        D = ones(dim, 1);
    else
        D = eye(dim, dim);
        I = eye(dim, dim);
        XX = X * X';
    end
    
    %% Initialize
    E1 = litekmeans(X', c);
    E = full(ind2vec(E1'))';
    Z = max(E, 0);
    B = orth(rand(d, c));
    W1 = eye(dim, d);
    F = X' * W1;
    W = eye(dim, c);
    
    obj = zeros(NITER, 1);
    
    for iter = 1:NITER
        %% Update B
        [Ub, ~, Vb] = svd(E' * F, 'econ');
        B = Vb * Ub';
        
        %% Update W
        if useSampleSpace
            rdiag = (pn / pa) * D + epsilon;
            if any(~isfinite(rdiag) | rdiag <= 0)
                error('DTAUFS:InvalidSampleRegularizer', ...
                      'Sample-space solver requires a positive finite regularization diagonal.');
            end
            % (X*X' + R)\(X*E) = R^(-1)*X*(I + X'*R^(-1)*X)^(-1)*E.
            RinvX = X ./ rdiag;
            W = RinvX * ((II + X' * RinvX) \ E);
        else
            W = (XX + (pn / pa) * D + epsilon * I) \ (X * E);
        end
        
        %% Update D
        dd = sqrt(sum(W .* W, 2) + eps);
        dd1 = (1 / 2) ./ dd;
        if useSampleSpace
            D = dd1;
        else
            D = diag(dd1);
        end
        
        %% Update E
        [Ue, ~, Ve] = svd(pa * W' * X + B' * F' + pd * Z', 'econ');
        E = Ve * Ue';
        
        %% Update Z
        Z = (E + abs(E)) / 2;
        
        %% Update F
        F = F_system \ (E * B');
        
        %% Objective Function
        obj(iter) = pa * norm(X' * W - E, 'fro')^2 + ...
                    norm(F' - B * E', 'fro')^2 + ...
                    pn * sum(dd) + ...
                    pd * norm(Z - E, 'fro')^2 + ...
                    pr * trace(F' * Ls * F);
        
        %% Convergence Check
        if (iter > 1 && abs((obj(iter) - obj(iter - 1)) / max(1, obj(iter - 1))) < 1e-6)
            obj = obj(1:iter); 
            break;
        end
    end
    
    [~, feaIdx] = sort(sum(W .* W, 2), 'descend');
end
