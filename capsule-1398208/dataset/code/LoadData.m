clear; clc; close all;
% 打开文件
filename = 'G:\dataset\data\01\BCG\01_20231104_BCG.csv';
fid = fopen(filename, 'rt');

% 读取第一行
firstLine = fgetl(fid);
firstLineData = textscan(firstLine, '%f', 'Delimiter', ',');

% 读取第二行
secondLine = fgetl(fid);
secondLineData = textscan(secondLine, '%f', 'Delimiter', ',');
secondVector = secondLineData{1};

timestamp = secondVector(2)
fs = secondVector(3)

data=readtable(filename);