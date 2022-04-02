function xmin=purecmaes   % (recombination_parents/mu_w, population_size)-CMA-ES
  % --------------------  Initialization --------------------------------
  % User defined input parameters (need to be edited)
  strfitnessfct = 'frosenbrock';  % name of objective/fitness function
  N = 20;               % number of objective variables/problem dimension
  xmean = rand(N,1);    % objective variables initial point
  sigma = 0.3;          % coordinate wise standard deviation (step size)
  stopfitness = 1e-10;  % stop if fitness < stopfitness (minimization)
  stopeval = 1e3*N^2;   % stop after stopeval number of function evaluations

  % Strategy parameter setting: Selection
  population_size = 4+floor(3*log(N));  % population size, offspring number
  recombination_parents = population_size/2;               % number of parents/points for recombination
  weights = log(recombination_parents+1/2)-log(1:recombination_parents)'; % muXone array for weighted recombination
  recombination_parents = floor(recombination_parents);
  weights = weights/sum(weights);     % normalize recombination weights array
  mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

  % Strategy parameter setting: Adaptation
  time_cumulation_c = (4+mueff/N) / (N+4 + 2*mueff/N);  % time constant for cumulation for covariance
  cumulation_for_sigma = (mueff+2) / (N+mueff+5);  % t-const for cumulation for sigma control
  learning_rate_for_c_update = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of covariance
  rank-recombination_parents_upgrade = min(1-learning_rate_for_c_update, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-recombination_parents update
  damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cumulation_for_sigma; % damping for sigma
                                                      % usually close to 1
  % Initialize dynamic (internal) strategy parameters and constants
  path_c = zeros(N,1); path_sigma = zeros(N,1);   % evolution paths for covariance and sigma
  coordinates = eye(N,N);                       % coordinates defines the coordinate system
  scaling = ones(N,1);                      % diagonal scaling defines the scaling
  covariance = coordinates * diag(scaling.^2) * coordinates';            % covariance matrix covariance
  inverse_sqrt_covariance = coordinates * diag(scaling.^-1) * coordinates';    % covariance^-1/2
  eigeneval = 0;                      % track update of coordinates and scaling
  chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of
                                      %   ||N(0,I)|| == norm(randn(N,1))
  % -------------------- Generation Loop --------------------------------
  counteval = 0;  % the next 40 lines contain the 20 lines of interesting code
  while counteval < stopeval

      % Generate and evaluate population_size offspring
      for k=1:population_size
          arx(:,k) = xmean + sigma * coordinates * (scaling .* randn(N,1)); % m + sig * Normal(0,covariance)
          arfitness(k) = feval(strfitnessfct, arx(:,k)); % objective function call
          counteval = counteval+1;
      end

      % Sort by fitness and compute weighted mean into xmean
      [arfitness, arindex] = sort(arfitness); % minimization
      mean_old = xmean;
      xmean = arx(:,arindex(1:recombination_parents))*weights;   % recombination, new mean value

      % Cumulation: Update evolution paths
      path_sigma = (1-cumulation_for_sigma)*path_sigma ...
            + sqrt(cumulation_for_sigma*(2-cumulation_for_sigma)*mueff) * inverse_sqrt_covariance * (xmean-mean_old) / sigma;
      hsig = norm(path_sigma)/sqrt(1-(1-cumulation_for_sigma)^(2*counteval/population_size))/chiN < 1.4 + 2/(N+1);
      path_c = (1-time_cumulation_c)*path_c ...
            + hsig * sqrt(time_cumulation_c*(2-time_cumulation_c)*mueff) * (xmean-mean_old) / sigma;

      % Adapt covariance matrix covariance
      artmp = (1/sigma) * (arx(:,arindex(1:recombination_parents))-repmat(mean_old,1,recombination_parents));
      covariance = (1-learning_rate_for_c_update-rank-recombination_parents_upgrade) * covariance ...                  % regard old matrix
           + learning_rate_for_c_update * (path_c*path_c' ...                 % plus rank one update
                   + (1-hsig) * time_cumulation_c*(2-time_cumulation_c) * covariance) ... % minor correction if hsig==0
           + rank-recombination_parents_upgrade * artmp * diag(weights) * artmp'; % plus rank recombination_parents update

      % Adapt step size sigma
      sigma = sigma * exp((cumulation_for_sigma/damps)*(norm(path_sigma)/chiN - 1));

      % Decomposition of covariance into coordinates*diag(scaling.^2)*coordinates' (diagonalization)
      if counteval - eigeneval > population_size/(learning_rate_for_c_update+rank-recombination_parents_upgrade)/N/10  % to achieve O(N^2)
          eigeneval = counteval;
          covariance = triu(covariance) + triu(covariance,1)'; % enforce symmetry
          [coordinates,scaling] = eig(covariance);           % eigen decomposition, coordinates==normalized eigenvectors
          scaling = sqrt(diag(scaling));        % scaling is a vector of standard deviations now
          inverse_sqrt_covariance = coordinates * diag(scaling.^-1) * coordinates';
      end

      % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable
      if arfitness(1) <= stopfitness || max(scaling) > 1e7 * min(scaling)
          break;
      end

  end % while, end generation loop

  xmin = arx(:, arindex(1)); % Return best point of last iteration.
                             % Notice that xmean is expected to be even
                             % better.
end
% ---------------------------------------------------------------
function f=frosenbrock(x)
    if size(x,1) < 2 error('dimension must be greater one'); end
    f = 100*sum((x(1:end-1).^2 - x(2:end)).^2) + sum((x(1:end-1)-1).^2);
end
