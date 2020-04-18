%�ó������ڽ��������ҵ������ȣ�m��������n����������nΪ��������������Ĺ���
%����������n���ӹ�������ΪM��ÿ��������ÿ��������ж����������ѡ�񣬶�Ӧ��ʱ��
%��ͬ�����г�ʼ��Ⱥ�Ĵ��淽ʽ����cell��������
%Version:1.3
%fileDescription:���Ȼ�����ѡ��������ҵ�������⣬����ͼ������,���Ƴ�ʼ�⼯�������ɢԭ�򣩣����ƽ��棨����ӳ�佻�棩,8*8ʵ��
%last edit time:2019-4-15
function main()
count = 10;     %��������
N = 50;          %��Ⱥ��ģ
pc = 0.8;       %��������
pm = 0.2;       %�������
m = 8;             %������
n = 4;             %������
M = 8;             %������
plotif = 1;        %���Ƴ����Ƿ���л�ͼ
s = input(m,n);    %��������
[p,TN] = initial_p(m,n,N,s,M);    %���ɳ�ʼ��Ⱥ50,����ϸ���ṹ��ÿ��Ԫ��Ϊ8*4
P = machine(n,M);
FIT = zeros(count,1);
aveFIT = zeros(count,1);
X1=randperm(count);       %����ͼ�εĺ�����X
X=sort(X1);
%------------------------������Ž��ʱ����------------------------------
best_fit = 1000;            %�ı�ģ����Ҫ�޸Ĵ˲���
best_p = zeros(m,n);
best_TN = zeros(m,n);
Y1p = zeros(m,1);
Y2p = zeros(m,1);
Y3p = zeros(m,1);
minfit3  =  1000000000;
%-------------------------���е���--------------------------------------
for i = 1:count
    [fit,Y1,Y2,Y3] = object(p,TN,N,P,m,n);
    [newp,newTN] = selection(p,TN,fit,N);
    [newp,newTN] = crossover(newp,pc,m,n,N,newTN);
    [newp,newTN] = var(newp,pm,m,n,N,s,newTN);
    if best_fit > min(fit)
        [best_p,best_TN,best_fit,Y1p,Y2p,Y3p]=best(best_fit,best_p,fit,best_TN,Y1p,Y2p,Y3p,p,TN,Y1,Y2,Y3);
    end
    p = newp;
    TN = newTN;
    minfit = min(fit);
    if minfit3>minfit
        minfit3 = minfit;
    end
    FIT(i) = minfit3;    %������Ӧ�Ⱥ�����
    aveFIT(i) = mean(fit);      %������Ӧ�Ⱥ�����
end
%------------------Ͷ����ѷ�������--------------------------------------
   
    fprintf('���Ž⣺%d\n',best_fit);
    fprintf('����1 ����2 ����3 ����4\n');
    best_p
    fprintf('ʱ��1 ʱ��2 ʱ��3 ʱ��4\n');
    best_TN
    Y1p
    Y2p
    Y3p
%------------------------��������----------------------------------------
    if plotif == 1
    figure;
    plot(X,FIT,'r');
    hold on;
    plot(X,aveFIT,'b');
    title('��������');
    hold on;
    legend('���Ž�','ƽ��ֵ');
%-------------------------����ͼ-----------------------------------------
figure;
w=0.5;       %������� 
set(gcf,'color','w');      %ͼ�ı�����Ϊ��ɫ
for i = 1:m
    for j = 1:n
        color=[1,0.98,0.98;1,0.89,0.71;0.86,0.86,0.86;0.38,0.72,1;1,0,1;0,1,1;0,1,0.49;1,0.87,0.67;0.39,0.58,0.92;0.56,0.73,0.56];
        a = [Y1p(i,j),Y2p(i,j)];
        x=a(1,[1 1 2 2]);      %����Сͼ���ĸ����x����
        y=Y3p(i,j)+[-w/2 w/2 w/2 -w/2];   %����Сͼ���ĸ����y����
        color = [color(i,1),color(i,2),color(i,3)];
        p=patch('xdata',x,'ydata',y,'facecolor',color,'edgecolor','k');    %facecolorΪ�����ɫ��edgecolorΪͼ����ɫ
            text(a(1,1)+0.5,Y3p(i,j),[num2str(i),'-',num2str(j)]);    %��ʾСͼ���������λ�ú���ֵ
    end
end
xlabel('�ӹ�ʱ��/s');      %����������
ylabel('����');            %����������
title({[num2str(m),'*',num2str(M),'��һ����ѵ��ȣ�����깤ʱ��Ϊ',num2str(best_fit),')']});      %ͼ������
axis([0,best_fit+2,0,M+1]);         %x�ᣬy��ķ�Χ
set(gca,'Box','on');       %��ʾͼ�α߿�
set(gca,'YTick',0:M+1);     %y�����������
set(gca,'YTickLabel',{'';num2str((1:M)','M%d');''});  %��ʾ������
hold on;
    end
%--------------------------��������---------------------------------
function s = input(m,n)      %��������
s = cell(m,n);
s{1,1}=[1 2 3 4 5 7 8;5 3 5 3 3 10 9];
s{1,2}=[1 3 4 5 6 7 8;10 5 8 3 9 9 6];
s{1,3}=[2 4 5 6 7 8;10 5 6 2 4 5];
s{1,4}=[0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0];
s{2,1}=[1 2 3 4 5 7;5 7 3 9 8 9];
s{2,2}=[2 3 4 5 6 7 8;8 5 2 6 7 10 9];
s{2,3}=[2 4 5 6 7 8;10 5 6 4 1 7];
s{2,4}=[1 2 3 4 5 6;10 8 9 6 4 7];

s{3,1}=[1 4 5 6 7 8;10 7 6 5 2 4];
s{3,2}=[2 3 4 5 6 7;10 6 4 8 9 10];
s{3,3}=[1 2 3 4 6 8;1 4 5 6 10 7];
s{3,4}=[0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0];
s{4,1}=[1 2 3 4 5 6 7 8;3 1 6 5 9 7 8 4];
s{4,2}=[1 2 3 4 5 6 7 8;12 11 7 8 10 5 6 9];
s{4,3}=[1 2 3 4 5 6 7 8;4 6 2 10 3 9 5 7];
s{4,4}=[0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0];
s{5,1}=[1 2 3 4 5 7;3 6 7 8 9 10];
s{5,2}=[1 3 4 5 6 7;10 7 4 9 8 6];
s{5,3}=[2 3 4 5 6 7;9 8 7 4 2 7];
s{5,4}=[1 2 4 5 6 7 8;11 9 6 7 5 3 6];

s{6,1}=[1 2 3 4 5 6 8;6 7 1 4 6 9 10];
s{6,2}=[1 3 4 5 6 7 8;11 9 9 9 7 6 4];
s{6,3}=[1 2 3 4 5 7;10 5 9 10 11 10];
s{6,4}=[0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0];
s{7,1}=[1 2 3 4 5 7;5 4 2 6 7 10];
s{7,2}=[2 4 5 6 7 8;9 9 11 9 10 5];
s{7,3}=[2 3 4 5 6 8;8 9 3 8 6 10];
s{7,4}=[0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0];
s{8,1}=[1 2 3 4 6 8;2 8 5 9 4 10];
s{8,2}=[1 2 3 4 5 7;7 4 7 8 9 10];
s{8,3}=[1 2 4 5 6 7 8;9 9 8 5 6 7 1];
s{8,4}=[1 3 4 5 6 7;9 3 7 1 5 8];  
%---------------------------������ʼ��Ⱥ-----------------------------
function [p,TN] = initial_p(m,n,N,s,M)     %������ʼ��Ⱥ
p = cell(N,1);            %pΪ��ʼ�⼯�Ļ�����
TN = cell(N,1);            %TNΪ��ʼ�⼯��ʱ�伯
for i = 1:N                  %����N����ʼ��
    store_m = zeros(M,1);    %���ڴ������ɳ�ʼ����ʱ�ĸ���������
    pz = zeros(m,n);         %pzΪ�м䴢���������ڴ����i�Ļ����ţ���СΪm*n
    tz = zeros(m,n);         %tzΪ�м䴢���������ڴ����i�ļӹ�ʱ�䣬��СΪm*n
    for j = 1:m
        for k = 1:n
            sle = s(j,k);       %sleΪ����j�Ĺ���k�����ݣ���һ��Ϊ��ѡ���������ڶ���Ϊ��Ӧ�ļӹ�ʱ��
            sle2 = cell2mat(sle);    %sleΪcell�ṹ����Ҫ��sle��cell2mat����ת��Ϊdouble����
            b = size(sle2,2);       %��������0���飬������Ҫ�ж�
            if b == 0
                pz(j,k) = 0;
                tz(j,k) = 0;
            else
            c = randperm(b,1);   %����һ��1��b�������������ѡ�����
                if store_m(c) >= (m*n)/M
                    c = randperm(b,1);
                        if store_m(c) >= (m*n)/M
                             c = randperm(b,1);
                             if store_m(c) >= (m*n)/M
                                c = randperm(b,1);
                             end
                        end
                end
            store_m(c) = store_m(c)+1;
            pz(j,k) = sle2(1,c);     %����������pz(j,k)
            tz(j,k) = sle2(2,c);     %���ӹ�ʱ�丳��tz(j,k)
            end
        end
    end
    p{i} = pz;
    TN{i} = tz;
end
%---------------------------����������������-----------------------
function P = machine(n,M)
P = zeros(n,1);
for i= 1:n
    P(i) = M;      %ÿ������Ŀ�ѡ��������ΪM
end
%-------------------------�����Ⱦɫ�����Ӧ��-----------------------
function [fit,Y1,Y2,Y3] = object(p,TN,N,P,m,n)  %�����Ⱦɫ�����Ӧ��
fit = zeros(N,1);
Y1 = cell(N,1);
Y2 = cell(N,1);
Y3 = cell(N,1);
    for j = 1:N
        Y1{j} = zeros(m,n);
        Y2{j} = zeros(m,n);
        Y3{j} = zeros(m,n);
    end
for w = 1:N
    X = p{w};                  %������ʼ��
    T = TN{w};
    [m,n] = size(X);
    Y1p = zeros(m,n);
    Y2p = zeros(m,n);
    Y3p = zeros(m,n);
    Q1 = zeros(m,1);         %�����һ������İ���
    Q2 = zeros(m,1);
    R = X(:,1);             %ȡ����һ������Ļ�����
    Q3 = floor(R);          %����ȡ���õ��������ڵ�һ������ʹ�õĻ�����
    for k =1:P(1)           %��һ�������ʱ�䰲�ţ�kΪ������
        pos = find(Q3 == k);     %��Q3��ȡ���û���k�ӹ��Ĺ������
        lenpos = length(pos);    %ʹ�û���k�Ĺ�������
        if lenpos == 0
        end
        if lenpos >= 1
            Q1(pos(1)) = 0;
            Q2(pos(1)) = Q1(pos(1)) + T(pos(1),1);
            if lenpos >= 2 
                for j = 2:lenpos
                    Q1(pos(j)) = Q2(pos(j-1));
                    Q2(pos(j)) = Q1(pos(j)) + T(pos(j),1);
                end
            end
        end
    end

    Y1p(:,1) = Q1;
    Y2p(:,1) = Q2;
    Y3p(:,1) = Q3;

    for k = 2:n            %�����2��n������İ���
        Q1 = zeros(m,1);
        Q2 = zeros(m,1);
        R = X(:,k);        %ȡ����k������Ļ�����
        Q3 = floor(R);     %����ȡ���õ��������ڵ�k������ʹ�õĻ�����
        R1 = X(:,k-1);     %ȡ��ǰһ������Ļ�����
        Q31 = floor(R1);   %����ȡ���õ���������ǰһ������ʹ�õĻ�����
        for i = 1:P(k)     %��i�������ʱ�䰲�ţ�kΪ������
            pos = find(Q3 == i);
            lenpos = length(pos);
            pos1 = find(Q31 == i);
            lenpos1 = length(pos1);
            if lenpos == 0
            end
            if lenpos >= 1
                EndTime = Y2p(pos(1),k-1);
                POS = zeros(1,lenpos1);
                for j = 1:lenpos1
                    POS(j) = Y2p(pos1(j),k-1);
                end
                EndTime1 = max(POS);
                if EndTime1 > EndTime
                    EndTime = EndTime1;
                else
                    EndTime = EndTime;
                end
                Q1(pos(1)) = EndTime;
                Q2(pos(1)) =  Q1(pos(1)) + T(pos(1),k-1);
                if lenpos >= 2
                    for j = 2:lenpos
                        Q1(pos(j)) = Y2p(pos(j),k-1);   %ǰһ������Ľ���ʱ��
                        if Q1(pos(j)) < Q2(pos(j-1))
                            Q1(pos(j)) = Q2(pos(j-1));
                        else
                             Q1(pos(j)) = Q1(pos(j));
                        end
                        Q2(pos(j)) = Q1(pos(j)) + T(pos(j),k);
                    end
                end
            end
        end
    Y1p(:,k) = Q1;
    Y2p(:,k) = Q2;
    Y3p(:,k) = Q3;
    end
    Y2m = Y2p(:,n);
    Y2m1 = Y2p(:,n-1);
    Zx = max(Y2m1);
    Zp = max(Y2m);
    if Zx >Zp
        Zp = Zx;
    end
    fit(w) = Zp;
    Y1{w} = Y1p;
    Y2{w} = Y2p;
    Y3{w} = Y3p;
end

%------------------------ѡ��N����Ӧ�ȸߵ�Ⱦɫ��---------------------
function [newp,newTN] = selection(p,TN,fit,N)
newp = cell(N,1);         %��������Ⱥ�Ĵ�С
newTN = cell(N,1);
a = zeros(N,1);
a(1:N) = 1;
fit = a./fit;
totalfit = sum(fit);      %��������Ӧ��ֵ�����ۼ�
p_fit = fit/totalfit;     %����ÿ��Ⱦɫ���ռ��
p_fit = cumsum(p_fit);    %�ۼ�
random = sort(rand(N,1));  %�������N��0-1����������������
fitin = 1;
newin = 1;
while newin <= N
    if (random(newin)) < p_fit(fitin)
        newp{newin} = p{fitin};
        newTN{newin} = TN{fitin};
        newin = newin + 1;
    else 
        fitin = fitin + 1;
    end
end

%------------------------��N��Ⱦɫ����н������---------------------
function [newp,newTN] = crossover(newp,pc,m,n,N,newTN)
for i = 1:2:N
    cross1 = newp{i};
    cross2 = newp{i+1};
    cross3 = newTN{i};
    cross4 = newTN{i+1};
    if(rand<pc)
        for j = 1:m
        	apoint = round(rand*n);
            cross1(m,:) = [cross1(m,1:apoint),cross2(m,apoint+1:n)];
            cross2(m,:) = [cross2(m,1:apoint),cross1(m,apoint+1:n)];
            cross3(m,:) = [cross3(m,1:apoint),cross4(m,apoint+1:n)];
            cross4(m,:) = [cross4(m,1:apoint),cross3(m,apoint+1:n)];      
        end
    end
    newp{i} = cross1;
    newp{i+1} = cross2;
    newTN{i} = cross3;
    newTN{i+1} = cross4;
end

%------------------------��N��Ⱦɫ����б������---------------------
function [newp,newTN] = var(newp,pm,m,n,N,s,newTN)
for i = 1:N
    if (rand<pm)
        bpoint = round(rand*m);
        cpoint = round(rand*n);
        if bpoint <= 0
            bpoint = 1;
        end
        if cpoint <= 0
            cpoint = 1;
        end
        var1 = newp{i};
        var2 = newTN{i};
        sle = s(bpoint,cpoint);
        sle2 = cell2mat(sle);
        [d,e] = size(sle2);
        epoint = round(rand*e);
        if epoint <= 0
            epoint = 1;
        end
        var1(bpoint,cpoint) = sle2(1,epoint);
        var2(bpoint,cpoint) = sle2(2,epoint);
        newp{i} = var1;
        newTN{i} = var2;
    else
        newp{i} = newp{i};
    end
end
%-----------------------------ѡ�����ŷ���---------------------------
function [best_p,best_TN,best_fit,Y1p,Y2p,Y3p]=best(best_fit,best_p,fit,best_TN,Y1p,Y2p,Y3p,p,TN,Y1,Y2,Y3)
    best_fit = min(fit);
    pos = find(fit==best_fit);
    best_p = p(pos(1));
    best_TN = TN(pos(1));
    best_p=cell2mat(best_p);
    best_TN=cell2mat(best_TN);
    Y1p=Y1(pos(1));
    Y2p=Y2(pos(1));
    Y3p=Y3(pos(1));
    Y1p=cell2mat(Y1p);
    Y2p=cell2mat(Y2p);
    Y3p=cell2mat(Y3p);



