close all;
clear all;
clc

zk_trj = readmatrix('mean_trj_bouncing.csv');
% cov_trj = readmatrix('cov_trj_bouncing.csv');
cov_trj_data = load('cov_trj_bouncing.mat');
sample_trj_data = load('sample_trj_bouncing.mat');

% Extract the matrix
cov_trj = cov_trj_data.matrix;
sample_trj = sample_trj_data.matrix;

nx = 2;
nt = size(cov_trj, 3);
% cov_trj = reshape(cov_trj, nt, nx, nx);
% cov_trj = permute(cov_trj, [2 3 1]);

m0 = [5; 1.5];
Sig0 = 0.2.*eye(nx);
zk_trj = zk_trj';

tf = 2.0;
plot_result = plot_cov_trj(zk_trj, cov_trj, nt, m0, Sig0, tf, sample_trj);