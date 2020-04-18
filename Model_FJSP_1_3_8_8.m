%该程序用于解决柔性作业车间调度，m个工件，n道工序，其中n为最大工序数，工件的工序
%数可以少于n，加工机器数为M，每个工件的每道工序具有多个机器可以选择，对应的时间
%不同，其中初始种群的储存方式采用cell数据类型
%Version:1.3
%fileDescription:调度机器可选的柔性作业车间问题，甘特图已完善,改善初始解集（均衡分散原则），改善交叉（部分映射交叉）,8*8实例
%last edit time:2019-4-15
function main()
count = 10;     %迭代次数
N = 50;          %种群规模
pc = 0.8;       %交换概率
pm = 0.2;       %变异概率
m = 8;             %工件数
n = 4;             %工序数
M = 8;             %机器数
plotif = 1;        %控制程序是否进行绘图
s = input(m,n);    %数据输入
[p,TN] = initial_p(m,n,N,s,M);    %生成初始种群50,采用细胞结构，每个元素为8*4
P = machine(n,M);
FIT = zeros(count,1);
aveFIT = zeros(count,1);
X1=randperm(count);       %收敛图形的横坐标X
X=sort(X1);
%------------------------输出最优解的时有用------------------------------
best_fit = 1000;            %改变模型需要修改此参数
best_p = zeros(m,n);
best_TN = zeros(m,n);
Y1p = zeros(m,1);
Y2p = zeros(m,1);
Y3p = zeros(m,1);
minfit3  =  1000000000;
%-------------------------进行迭代--------------------------------------
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
    FIT(i) = minfit3;    %用于适应度函数的
    aveFIT(i) = mean(fit);      %用于适应度函数的
end
%------------------投射最佳方案数据--------------------------------------
   
    fprintf('最优解：%d\n',best_fit);
    fprintf('工序1 工序2 工序3 工序4\n');
    best_p
    fprintf('时间1 时间2 时间3 时间4\n');
    best_TN
    Y1p
    Y2p
    Y3p
%------------------------收敛曲线----------------------------------------
    if plotif == 1
    figure;
    plot(X,FIT,'r');
    hold on;
    plot(X,aveFIT,'b');
    title('收敛曲线');
    hold on;
    legend('最优解','平均值');
%-------------------------甘特图-----------------------------------------
figure;
w=0.5;       %横条宽度 
set(gcf,'color','w');      %图的背景设为白色
for i = 1:m
    for j = 1:n
        color=[1,0.98,0.98;1,0.89,0.71;0.86,0.86,0.86;0.38,0.72,1;1,0,1;0,1,1;0,1,0.49;1,0.87,0.67;0.39,0.58,0.92;0.56,0.73,0.56];
        a = [Y1p(i,j),Y2p(i,j)];
        x=a(1,[1 1 2 2]);      %设置小图框四个点的x坐标
        y=Y3p(i,j)+[-w/2 w/2 w/2 -w/2];   %设置小图框四个点的y坐标
        color = [color(i,1),color(i,2),color(i,3)];
        p=patch('xdata',x,'ydata',y,'facecolor',color,'edgecolor','k');    %facecolor为填充颜色，edgecolor为图框颜色
            text(a(1,1)+0.5,Y3p(i,j),[num2str(i),'-',num2str(j)]);    %显示小图框里的数字位置和数值
    end
end
xlabel('加工时间/s');      %横坐标名称
ylabel('机器');            %纵坐标名称
title({[num2str(m),'*',num2str(M),'的一个最佳调度（最短完工时间为',num2str(best_fit),')']});      %图形名称
axis([0,best_fit+2,0,M+1]);         %x轴，y轴的范围
set(gca,'Box','on');       %显示图形边框
set(gca,'YTick',0:M+1);     %y轴的增长幅度
set(gca,'YTickLabel',{'';num2str((1:M)','M%d');''});  %显示机器号
hold on;
    end
%--------------------------输入数据---------------------------------
function s = input(m,n)      %输入数据
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
%---------------------------建立初始种群-----------------------------
function [p,TN] = initial_p(m,n,N,s,M)     %建立初始种群
p = cell(N,1);            %p为初始解集的机器集
TN = cell(N,1);            %TN为初始解集的时间集
for i = 1:N                  %产生N个初始解
    store_m = zeros(M,1);    %用于储存生成初始方案时的各机器数量
    pz = zeros(m,n);         %pz为中间储存量，用于储存解i的机器号，大小为m*n
    tz = zeros(m,n);         %tz为中间储存量，用于储存解i的加工时间，大小为m*n
    for j = 1:m
        for k = 1:n
            sle = s(j,k);       %sle为工件j的工序k的数据，第一行为可选机器数，第二行为对应的加工时间
            sle2 = cell2mat(sle);    %sle为cell结构，需要将sle用cell2mat函数转换为double类型
            b = size(sle2,2);       %数据中有0数组，所以需要判断
            if b == 0
                pz(j,k) = 0;
                tz(j,k) = 0;
            else
            c = randperm(b,1);   %产生一个1到b的随机数，用于选择机器
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
            pz(j,k) = sle2(1,c);     %将机器赋予pz(j,k)
            tz(j,k) = sle2(2,c);     %将加工时间赋予tz(j,k)
            end
        end
    end
    p{i} = pz;
    TN{i} = tz;
end
%---------------------------输入各工序机器数量-----------------------
function P = machine(n,M)
P = zeros(n,1);
for i= 1:n
    P(i) = M;      %每道工序的可选机器数设为M
end
%-------------------------计算各染色体的适应度-----------------------
function [fit,Y1,Y2,Y3] = object(p,TN,N,P,m,n)  %计算各染色体的适应度
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
    X = p{w};                  %变量初始化
    T = TN{w};
    [m,n] = size(X);
    Y1p = zeros(m,n);
    Y2p = zeros(m,n);
    Y3p = zeros(m,n);
    Q1 = zeros(m,1);         %计算第一道工序的安排
    Q2 = zeros(m,1);
    R = X(:,1);             %取出第一道工序的机器号
    Q3 = floor(R);          %向下取整得到各工件在第一道工序使用的机器号
    for k =1:P(1)           %第一道工序的时间安排，k为机器号
        pos = find(Q3 == k);     %在Q3中取出用机器k加工的工件编号
        lenpos = length(pos);    %使用机器k的工件数量
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

    for k = 2:n            %计算第2到n道工序的安排
        Q1 = zeros(m,1);
        Q2 = zeros(m,1);
        R = X(:,k);        %取出第k道工序的机器号
        Q3 = floor(R);     %向下取整得到各工件在第k道工序使用的机器号
        R1 = X(:,k-1);     %取出前一道工序的机器号
        Q31 = floor(R1);   %向下取整得到各工件在前一道工序使用的机器号
        for i = 1:P(k)     %第i道工序的时间安排，k为机器号
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
                        Q1(pos(j)) = Y2p(pos(j),k-1);   %前一道工序的结束时间
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

%------------------------选择N个适应度高的染色体---------------------
function [newp,newTN] = selection(p,TN,fit,N)
newp = cell(N,1);         %定义新种群的大小
newTN = cell(N,1);
a = zeros(N,1);
a(1:N) = 1;
fit = a./fit;
totalfit = sum(fit);      %将所有适应度值进行累加
p_fit = fit/totalfit;     %计算每个染色体的占比
p_fit = cumsum(p_fit);    %累加
random = sort(rand(N,1));  %随机生成N个0-1的数，并降序排列
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

%------------------------对N个染色体进行交叉操作---------------------
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

%------------------------对N个染色体进行变异操作---------------------
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
%-----------------------------选择最优方案---------------------------
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



