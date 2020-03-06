function [Q1]=modul(A,a)
one = 1:length(A);
a = [one' a];
% 　建立节点社区矩阵
a = accumarray(a,1);
a = a(:,any(a));%　 删除A中全0的列
% 　进行网络A模块度Ｑ1运算
m = sum(sum(A))/2;
k = sum(A,2);
tmp = (repmat(k,[1,size(A,1)]) .* repmat(k',[size(A,1),1])) / (2*m);
B = A - tmp;
Q1 = 1/(2*m) .* trace(a'*B*a);
end
