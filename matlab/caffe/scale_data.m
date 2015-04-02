function y = scale_data(x, Imin, Imax)
% SCALE  Scales matrix elements to a new range.
%
%   y = SCALE(x,[min max]) scales the elements of matrix x to a new range
%   defined by [min, max].
%   
%   y = SCALE(x) uses the default range = [0 1]
%
M = max(x(:));
m = min(x(:));
y = ( (x-m)/(M-m) * (Imax-Imin) ) + Imin;
