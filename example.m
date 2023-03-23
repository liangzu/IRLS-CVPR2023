clc; clear all; format longG

%% parameters
m = 1000; 
k = 900;

sigma = 0.01; % noise

noise_bound =5.54*sigma;

stopTh = 1e-10;

%% generate data 

problem.N                = m;
problem.outlierRatio     = k/m;
problem.noiseSigma       = sigma;
problem.translationBound = 10.0;
problem                  = gen_point_cloud_registration(problem);
X = problem.cloudA; 
Y = problem.cloudB;
R_gt = problem.R_gt;
t_gt = problem.t_gt;

%% run algorithms

% run MS-GNC-TLS
[mgnc.R, mgnc.t] = GNC_TLS_PointCloudRegistration(Y, X, noise_bound^2, stopTh, 1, 1);


% run GNC-IRLS0
epsilon = 1; beta = 0.8;

[IRLS0.R, IRLS0.t] = GNC_IRLS0_PointCloudRegistration(Y, X, epsilon, beta, noise_bound, stopTh);

%% errors
[getAngularError(R_gt,mgnc.R) getAngularError(R_gt, IRLS0.R)]

[norm(mgnc.t - t_gt) / norm(t_gt) norm(IRLS0.t - t_gt) / norm(t_gt)]


%% IMPLEMENTATION DETAILS

%% GNC-IRLS0
function [R_hat, t_hat] = GNC_IRLS0_PointCloudRegistration(Y, X, epsilon, beta, epsilon_min, stopTh)
% Y: 3xm
% X: 3xm
    [~, m] = size(X);
    weights = ones(1, m);
    
    pre_cost = inf;
    
    for i = 1:100
      [R_hat,t_hat] = LS_PointCloudRegistration(Y, X, weights);   

      D = Y - R_hat * X - t_hat;  

      residual = vecnorm(D);                         

      weights = 1./max(residual, epsilon);
      weights = weights.^2;
      
      cost = weights * (residual').^2;  
      
      cost_diff = abs(cost - pre_cost);

      if cost_diff <= stopTh 
          break;
      end
         
      epsilon = max(beta*epsilon^2, epsilon_min);    
      pre_cost = cost;
    end      
end



%% (MS-)GNC-TLS
function [R_hat, t_hat] = GNC_TLS_PointCloudRegistration(Y, X, noise_bound_squared, stopTh, majorize, superlinear)
    [~, m] = size(Y);
    
    weights = ones(1,m);
    
    pre_cost = inf;
         
    for i=1:100
        [R_hat, t_hat] = LS_PointCloudRegistration(Y, X, weights);  
        
        abs_residual = vecnorm(Y - R_hat*X - t_hat);

        squared_residual = abs_residual.^2;
        
        max_residual = max(squared_residual);
        
        if i == 1
            if majorize
                mu = max(1 / (5*max_residual/noise_bound_squared - 1), 1e-6);
            else
                mu = max(1 / (5*max_residual/noise_bound_squared - 1), 1e-6);
            end
        end
      
        if majorize
            [weights] = M_gncWeightsUpdate(weights, mu, squared_residual, noise_bound_squared);                         
        else
            [weights] = gncWeightsUpdate(weights, mu, squared_residual, noise_bound_squared);                         
        end
        
        cost = weights * squared_residual';   
              
        cost_diff = abs(cost - pre_cost);
        
        if cost_diff <= stopTh 
            break;
        end
   
        if superlinear
            if mu < 1
                mu = min(sqrt(mu) * 1.4, 1e16);
            else
                mu = min(mu * 1.4, 1e16);
            end
        else
            mu = min(mu * 1.4, 1e16);
        end
        pre_cost = cost;
    end
end

%% Other help functions
function rotError = getAngularError(R_gt,R_est)

rotError = abs(acos( (trace(R_gt' * R_est)-1) / 2 ));
rotError = rad2deg( rotError );
end

function [R, t] = LS_PointCloudRegistration(Y, X, weights)
    s = sum(weights);

    x = X * weights' / s;
    y = Y * weights' / s;

    X_ = X - x; Y_ = Y - y;

    sqrtw = sqrt(weights);
    R= LS_RotationSearch( sqrtw .* Y_, sqrtw .* X_);  

    t = y - R*x;
end

function [R_hat] = LS_RotationSearch(Y, X)
    % Y = R*X
    % Y: 3xm 
    % X: 3xm
    M = Y*X';

    [U, ~, V] = svd(M);

    D = diag([1,1, det(U)*det(V)]);

    R_hat = U*D*V';
end

function [weights] = M_gncWeightsUpdate(weights, mu, residuals, barc2)

    ub = (mu+1)^2/mu^2 * barc2;
    lb = barc2; % 

    for k = 1:length(residuals)
        if residuals(k) - ub >= 0
            weights(k) = 0;
        elseif residuals(k) - lb <= 0
            weights(k) = 1;
        else
            weights(k) = sqrt( barc2/residuals(k) )*(mu+1)  - mu;
        end
    end
end


function [weights, lb, ub] = gncWeightsUpdate(weights, mu, residuals, barc2)
ub = (mu+1)/mu * barc2;
lb = (mu)/(mu+1) * barc2; 

for k = 1:length(residuals)
    if residuals(k) - ub >= 0
        weights(k) = 0;
        
    elseif residuals(k) - lb <= 0
        weights(k) = 1;
        
    else                
        weights(k) = sqrt( barc2*mu*(mu+1)/residuals(k) ) - mu;
    end
end
end

% Code taken from: https://github.com/MIT-SPARK/CertifiablyRobustPerception/blob/master/PointCloudRegistration/solvers/gen_point_cloud_registration.m
function problem = gen_point_cloud_registration(problem)
%% Generate random point cloud registration problem
%% Heng Yang
%% June 25, 2021

if ~isfield(problem, 'N'); error('Please use problem.N to specify the number of correspondences.'); end
if ~isfield(problem, 'outlierRatio'); problem.outlierRatio = 0.0; end
if ~isfield(problem, 'translationBound'); problem.translationBound = 10.0; end
if ~isfield(problem, 'noiseSigma'); problem.noiseSigma = 0.01; end

N                   = problem.N;
outlierRatio        = problem.outlierRatio;
translationBound    = problem.translationBound;
noiseSigma          = problem.noiseSigma;

% random point cloud A
cloudA              = randn(3,N);
% random ground-truth transformation
R_gt                = rand_rotation;
t_gt                = randn(3,1);
t_gt                = t_gt/norm(t_gt); 
t_gt                = (translationBound) * rand * t_gt;
% point cloud B, transformed and add noise
cloudB              = R_gt * cloudA + t_gt + noiseSigma * randn(3,N);
% add outliers 
nrOutliers          = round(N * outlierRatio);
if (N - nrOutliers) < 3
    error('Point cloud registration requires minimum 3 inlier correspondences.')
end

if nrOutliers > 0
    fprintf('point cloud registration: random generate %d outliers.\n',nrOutliers)
    outlierB        = randn(3,nrOutliers);
    center_B        = mean(cloudB,2);
    outlierIDs      = N-nrOutliers+1:N;
    cloudB(:,outlierIDs) = outlierB + center_B;
else
    outlierIDs = [];
end
% add data to the problem structure
problem.type        = 'point cloud registration';
problem.cloudA      = cloudA;
problem.cloudB      = cloudB;
problem.nrOutliers  = nrOutliers;
problem.outlierIDs  = outlierIDs;
problem.R_gt        = R_gt;
problem.t_gt        = t_gt;
% note that the noiseBoundSq is important to ensure tight relaxation and
% good numerical performance of the solvers. If noiseBound is too small,
% typically the SDP solvers will perform worse (especially SDPNAL+)
noiseBoundSq        = noiseSigma^2 * chi2inv(0.99,3);
noiseBoundSq        = max(4e-2,noiseBoundSq); 
problem.noiseBoundSq= noiseBoundSq;
problem.noiseBound  = sqrt(problem.noiseBoundSq);

fprintf('N: %d, outlierRatio: %g, translationBound: %g, noiseBoundSq: %g, noiseBound: %g.\n',...
    N,outlierRatio,translationBound,noiseBoundSq,problem.noiseBound);
end

function R = rand_rotation(varargin)
% generate random rotation matrix

params = inputParser;
params.CaseSensitive = false;

params.addParameter('RotationBound',2*pi,...
    @(x) 0.0<=x && x<=2*pi);

params.parse(varargin{:});

RotationBound = params.Results.RotationBound;

angle = RotationBound*rand - RotationBound/2;
axis  = randn(3,1);
axis  = axis / norm(axis);
R     = axang2rotm([axis' angle]);
end
