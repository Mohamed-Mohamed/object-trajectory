%% Coded by
% Mohamed Mohamed El-Sayed Atyya
% mohamed.atyya94@eng-st.cu.edu.eg


% The targt of this program is:
% The target is to plot velocity vector, acceleration vector, position vector, tangential acceleration
% vector, normal acceleration vector, radius of curvature vector, velocity vector and radius of curvature vector plane 
% and velocity vector and position vector plane at each point on the trajectory
close all; clear all; clc;
syms t; % function paramter
on=1; % variable used in code don't change it
%% Inputs
hz=1E2;  % animation hz
t_initial=0; % initial parameter value
t_final=5; % final parameter value
steps=100; % no. of steps
%% Time period of solution
T=linspace(t_initial,t_final,steps);
%% position function
x(t)=t*sin(t);
y(t)=t*cos(t);
z(t)=3*t;
%% velocity
vx(t)=simplify(diff(x(t)));
vy(t)=simplify(diff(y(t)));
vz(t)=simplify(diff(z(t)));
MagV(t)=simplify(sqrt(vx(t).^2+vy(t).^2+vz(t).^2));
%% acceleration
ax(t)=simplify(diff(vx(t)));
ay(t)=simplify(diff(vy(t)));
az(t)=simplify(diff(vz(t)));
MagA(t)=simplify(sqrt(ax(t).^2+ay(t).^2+az(t).^2));
%% unit vectors
% ut
utx(t)=simplify(vx(t)/MagV(t));
uty(t)=simplify(vy(t)/MagV(t));
utz(t)=simplify(vz(t)/MagV(t));
% ub
VcrossAx(t)=simplify(vy(t)*az(t)-vz(t)*ay(t));
VcrossAy(t)=simplify(vz(t)*ax(t)-vx(t)*az(t));
VcrossAz(t)=simplify(vx(t)*ay(t)-vy(t)*ax(t));
MagVcrossA(t)=simplify(sqrt(VcrossAx(t).^2+VcrossAy(t).^2+VcrossAz(t).^2));
ubx(t)=simplify(VcrossAx(t)/MagVcrossA(t));
uby(t)=simplify(VcrossAy(t)/MagVcrossA(t));
ubz(t)=simplify(VcrossAz(t)/MagVcrossA(t));
% un
unx(t)=simplify(uby(t)*utz(t)-ubz(t)*uty(t));
uny(t)=simplify(ubz(t)*utx(t)-ubx(t)*utz(t));
unz(t)=simplify(ubx(t)*uty(t)-uby(t)*utx(t));
%% normal and tangetial acceleration
at(t)=simplify(ax(t)*utx(t)+ay(t)*uty(t)+az(t)*utz(t));
an(t)=simplify(ax(t)*unx(t)+ay(t)*uny(t)+az(t)*unz(t));
%% raduis of curvture
c(t)=simplify(MagV(t).^2/an(t));
%% plotting
% position, velocity and acceleration at any point at time (t)
for i = 1:length(T)
    X(i)=double(x(T(i)));
    Y(i)=double(y(T(i)));
    Z(i)=double(z(T(i)));
    VX(i)=double(vx(T(i)));
    VY(i)=double(vy(T(i)));
    VZ(i)=double(vz(T(i)));
    AX(i)=double(ax(T(i)));
    AY(i)=double(ay(T(i)));
    AZ(i)=double(az(T(i)));
    C(i)=double(c(T(i)));
    Unx(i)=double(unx(T(i)));
    Uny(i)=double(uny(T(i)));
    Unz(i)=double(unz(T(i)));
end
% change parameter t
% Blue color indicates to get solution at any point of the parameter t in separate figures.
% Green color indicates to start animation from this parameter(t) value in separate figures.
% Yellow color indicates to get solution at any point of the parameter t in one figure.
% Cyan color indicates to start animation from this parameter(t) value in one figure.
% Red color indicates to stop the program (NOTE: you must do this if you to close figures).
figure(1);
set(gcf,'Color','w')
hold all;
area([0,0,T(end),T(end)],[0,0.5,0.5,0],'FaceColor','blue');
area([0,T(end),T(end),0],[1,1,0.5,0.5],'FaceColor','green');
area([0,T(end),T(end),0],[1.5,1.5,1,1],'FaceColor','yellow');
area([0,T(end),T(end),0],[2,2,1.5,1.5],'FaceColor','cyan');
area([0,T(end),T(end),0],[2.5,2.5,2,2],'FaceColor','red');
grid on;
xlim([0,T(end)]);
ylim([0,2.5]);
title('Change Parameter','Fontsize',18);
xlabel('Parameter','Fontsize',20);
legend('Parameter Change','Animation','Parameter Change in Full Plot','Animation in Full Plot','Close');
% absolute velocity, absolute acceleration, position and raduis of cuevature
figure(2);
set(gcf,'Color','w')
for j = 1:length(T)
    plot3(X,Y,Z,'color','blue','LineWidth',2); % curve
    hold on;
    plot3(X+C.*Unx,Y+C.*Uny,Z+C.*Unz,'color',[0.5,0.25,0],'LineWidth',2); % raduis of curvature curve
    hold on;
    xlim auto; ylim auto; zlim auto;
    title('Point Mass Moving on Trajectroy','Fontsize',18);
    xlabel('X','Fontsize',20);
    ylabel('Y','Fontsize',20);
    zlabel('Z','Fontsize',20);
    vectarrow([X(j),Y(j),Z(j)],[X(j)+VX(j),Y(j)+VY(j),Z(j)+VZ(j)],2,'green'); % velocity
    vectarrow([X(j),Y(j),Z(j)],[X(j)+AX(j),Y(j)+AY(j),Z(j)+AZ(j)],2,'red'); % acceleration
    vectarrow([0,0,0],[X(j),Y(j),Z(j)],2,'black'); % position
    vectarrow([X(j)+double(c(T(j))*unx(T(j))),Y(j)+double(c(T(j))*uny(T(j))),Z(j)+double(c(T(j))*unz(T(j)))],[X(j),Y(j),Z(j)],2,'cyan'); % raduis of curvature
    view([acosd(double(unx(T(j)))),acosd(double(uny(T(j)))),acosd(double(unz(T(j))))]);
    vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
    vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
    vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
    grid on;
    legend('Trajectory','Raduis of curvature curve','Velocity','Acceleration','Position','Raduis Of Curvature');
    pause(1/hz);
    hold off;
end
% absolute velocity, normal acceleration, tangetial acceleration, position and raduis of cuevature
figure(3);
set(gcf,'Color','w')
for I = 1:length(T)
    plot3(X,Y,Z,'blue','LineWidth',2); % curve
    hold on;
    xlim auto; ylim auto; zlim auto;
    vectarrow([0,0,0],[X(I),Y(I),Z(I)],2,'black'); % position
    vectarrow([X(I),Y(I),Z(I)],[X(I)+VX(I),Y(I)+VY(I),Z(I)+VZ(I)],2,'green') % velocity
    vectarrow([X(I),Y(I),Z(I)],[X(I)+double(an(T(I))*unx(T(I))),Y(I)+double(an(T(I))*uny(T(I))),Z(I)+double(an(T(I))*unz(T(I)))],2,[0.7,0.2,0.5]); % normal acceleration
    vectarrow([X(I),Y(I),Z(I)],[X(I)+double(at(T(I))*utx(T(I))),Y(I)+double(at(T(I))*uty(T(I))),Z(I)+double(at(T(I))*utz(T(I)))],2,'red'); % tangential acceleration
    vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
    vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
    vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
    view([acosd(double(unx(T(I)))),acosd(double(uny(T(I)))),acosd(double(unz(T(I))))]);
    grid on;
    title('Point Mass Moving on Trajectroy','Fontsize',18);
    xlabel('X','Fontsize',20);
    ylabel('Y','Fontsize',20);
    zlabel('Z','Fontsize',20);
    legend('Trajectory','Position','Velocity','Normal Acceleration','Tangential Acceleration');
    pause(1/hz);
    hold off;
end
% Velocity Position Plane and Velocity Raduis Of Cuevature Plane
figure(4);
set(gcf,'Color','w')
for J = 1:length(T)
    plot3(X,Y,Z,'blue','LineWidth',2); % curve
    hold on;
    xlim auto; ylim auto; zlim auto;
    % velocity position plane
    pointA = [X(J),Y(J),Z(J)];
    pointB = [X(J)+VX(J),Y(J)+VY(J),Z(J)+VZ(J)];
    pointC = [0,0,0];
    plane1 = cross(pointA-pointB, pointA-pointC);
    points=[pointA' pointB' pointC']; % using the data given in the question
    fill3(points(1,:),points(2,:),points(3,:),'r')
    alpha(0.5)
    % velocity raduis of cuevature plane
    pointA1 = [X(J),Y(J),Z(J)];
    pointB1 = [X(J)+VX(J),Y(J)+VY(J),Z(J)+VZ(J)];
    pointC1 = [X(J)+double(c(T(J))*unx(T(J))),Y(J)+double(c(T(J))*uny(T(J))),Z(J)+double(c(T(J))*unz(T(J)))];
    plane11 = cross(pointA1-pointB1, pointA1-pointC1);
    points1=[pointA1' pointB1' pointC1']; % using the data given in the question
    fill3(points1(1,:),points1(2,:),points1(3,:),'g')
    alpha(0.5)
    % velocity position plane
    pointAd = [VX(J),VY(J),VZ(J)];
    pointBd = [X(J)+VX(J),Y(J)+VY(J),Z(J)+VZ(J)];
    pointCd = [0,0,0];
    plane1d = cross(pointAd-pointBd, pointAd-pointCd);
    pointsd=[pointAd' pointBd' pointCd']; % using the data given in the question
    fill3(pointsd(1,:),pointsd(2,:),pointsd(3,:),'r')
    alpha(0.5)
    % velocity raduis of cuevature plane
    pointA1 = [X(J)+VX(J)+double(c(T(J))*unx(T(J))),Y(J)+VY(J)+double(c(T(J))*uny(T(J))),Z(J)+VZ(J)+double(c(T(J))*unz(T(J)))];
    pointB1 = [X(J)+VX(J),Y(J)+VY(J),Z(J)+VZ(J)];
    pointC1 = [X(J)+double(c(T(J))*unx(T(J))),Y(J)+double(c(T(J))*uny(T(J))),Z(J)+double(c(T(J))*unz(T(J)))];
    plane11 = cross(pointA1-pointB1, pointA1-pointC1);
    points1=[pointA1' pointB1' pointC1']; % using the data given in the question
    fill3(points1(1,:),points1(2,:),points1(3,:),'g')
    alpha(0.5)
    vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
    vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
    vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
    view([acosd(double(unx(T(J)))),acosd(double(uny(T(J)))),acosd(double(unz(T(J))))]);
    grid on;
    title('Point Mass Moving on Trajectroy','Fontsize',18);
    xlabel('X','Fontsize',20);
    ylabel('Y','Fontsize',20);
    zlabel('Z','Fontsize',20);
    legend('Trajectory','Velocity Position Plane','Velocity Raduis Of Cuevature Plane');
    pause(1/hz);
    hold off;
end
%% change parameter t
while (figure(1) && on == 1)
    [n,m] = ginput(1);
    paramerter_t=n
    if (n > 0 && n < T(end) && m > 0 && m < 0.5)
        % absolute velocity, absolute acceleration, position and raduis of cuevature
        figure(2);
        set(gcf,'Color','w')
        plot3(X,Y,Z,'blue','LineWidth',2); % curve
        hold on;
        plot3(X+C.*Unx,Y+C.*Uny,Z+C.*Unz,'color',[0.5,0.25,0],'LineWidth',2); % raduis of curvature curve
        hold on;
        xlim auto; ylim auto; zlim auto;
        title('Point Mass Moving on Trajectroy','Fontsize',18);
        xlabel('X','Fontsize',20);
        ylabel('Y','Fontsize',20);
        zlabel('Z','Fontsize',20);
        vectarrow([double(x(n)),double(y(n)),double(z(n))],[double(x(n))+double(vx(n)),double(y(n))+double(vy(n)),double(z(n))+double(vz(n))],2,'green'); % velocity
        vectarrow(double([x(n),y(n),z(n)]),double([x(n)+ax(n),y(n)+ay(n),z(n)+az(n)]),2,'red'); % acceleration
        vectarrow([0,0,0],double([x(n),y(n),z(n)]),2,'black'); % position
        vectarrow([double(x(n)+c(n)*unx(n)),double(y(n)+c(n)*uny(n)),double(z(n)+c(n)*unz(n))],double([x(n),y(n),z(n)]),2,'cyan'); % raduis of curvature
        vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
        vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
        vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
        view([-acosd(double(utx(n))),-acosd(double(uty(n))),acosd(double(utz(n)))]);
        grid on;
        legend('Trajectory','Raduis of curvature curve','Velocity','Acceleration','Position','Raduis Of Curvature');
        hold off;
        % absolute velocity, normal acceleration, tangetial acceleration, position and raduis of cuevature
        figure(3);
        set(gcf,'Color','w')
        plot3(X,Y,Z,'blue','LineWidth',2); % curve
        hold on;
        xlim auto; ylim auto; zlim auto;
        vectarrow([0,0,0],double([x(n),y(n),z(n)]),2,'black'); % position
        vectarrow(double([x(n),y(n),z(n)]),double([x(n)+vx(n),y(n)+vy(n),z(n)+vz(n)]),2,[0.25,0.25,0.5]) % velocity
        vectarrow(double([x(n),y(n),z(n)]),double([x(n)+an(n)*unx(n),y(n)+an(n)*uny(n),z(n)+an(n)*unz(n)]),2,[0.7,0.2,0.5]); % normal acceleration
        vectarrow(double([x(n),y(n),z(n)]),double([x(n)+at(n)*utx(n),y(n)+at(n)*uty(n),z(n)+at(n)*utz(n)]),2,'red'); % tangential acceleration
        vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
        vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
        vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
        view([-acosd(double(utx(n))),-acosd(double(uty(n))),acosd(double(utz(n)))]);
        grid on;
        title('Point Mass Moving on Trajectroy','Fontsize',18);
        xlabel('X','Fontsize',20);
        ylabel('Y','Fontsize',20);
        zlabel('Z','Fontsize',20);
        legend('Trajectory','Position','Velocity','Normal Acceleration','Tangential Acceleration');
        hold off;
        % Velocity Position Plane and Velocity Raduis Of Cuevature Plane
        figure(4);
        set(gcf,'Color','w')
        plot3(X,Y,Z,'blue','LineWidth',2); % curve
        hold on;
        xlim auto; ylim auto; zlim auto;
        % velocity position plane
        pointA = double([x(n),y(n),z(n)]);
        pointB = double([x(n)+vx(n),y(n)+vy(n),z(n)+vz(n)]);
        pointC = [0,0,0];
        plane1 = double(cross(pointA-pointB, pointA-pointC));
        points=[pointA' pointB' pointC']; % using the data given in the question
        fill3(points(1,:),points(2,:),points(3,:),'r')
        alpha(0.5)
        % velocity raduis of cuevature plane
        pointA1 = double([x(n),y(n),z(n)]);
        pointB1 = double([x(n)+vx(n),y(n)+vy(n),z(n)+vz(n)]);
        pointC1 = double([x(n)+double(c(n)*unx(n)),y(n)+double(c(n)*uny(n)),z(n)+double(c(n)*unz(n))]);
        plane11 = double(cross(pointA1-pointB1, pointA1-pointC1));
        points1=[pointA1' pointB1' pointC1']; % using the data given in the question
        fill3(points1(1,:),points1(2,:),points1(3,:),'g')
        alpha(0.5)
        vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
        vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
        vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
        view([-acosd(double(utx(n))),-acosd(double(uty(n))),acosd(double(utz(n)))]);
        grid on;
        title('Point Mass Moving on Trajectroy','Fontsize',18);
        xlabel('X','Fontsize',20);
        ylabel('Y','Fontsize',20);
        zlabel('Z','Fontsize',20);
        legend('Trajectory','Velocity Position Plane','Velocity Raduis Of Cuevature Plane');
        hold off;
    elseif (n > 0 && n < T(end) && m > 0.5 && m < 1)
        S=steps-n/atan((t_final-t_initial)/steps); % no. of steps
        P=linspace(n,t_final,S);
        for h=1:length(P)
            % absolute velocity, absolute acceleration, position and raduis of cuevature
            figure(2);
            set(gcf,'Color','w')
            plot3(X,Y,Z,'blue','LineWidth',2); % curve
            hold on;
            plot3(X+C.*Unx,Y+C.*Uny,Z+C.*Unz,'color',[0.5,0.25,0],'LineWidth',2); % raduis of curvature curve
            hold on;
            xlim auto; ylim auto; zlim auto;
            title('Point Mass Moving on Trajectroy','Fontsize',18);
            xlabel('X','Fontsize',20);
            ylabel('Y','Fontsize',20);
            zlabel('Z','Fontsize',20);
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+vx(P(h)),y(P(h))+vy(P(h)),z(P(h))+vz(P(h))]),2,'green'); % velocity
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+ax(P(h)),y(P(h))+ay(P(h)),z(P(h))+az(P(h))]),2,'red'); % acceleration
            vectarrow([0,0,0],double([x(P(h)),y(P(h)),z(P(h))]),2,'black'); % position
            vectarrow(double([x(P(h))+c(P(h))*unx(P(h)),y(P(h))+c(P(h))*uny(P(h)),z(P(h))+c(P(h))*unz(P(h))]),double([x(P(h)),y(P(h)),z(P(h))]),2,'cyan'); % raduis of curvature
            vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
            vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
            vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
            view([-acosd(double(utx(P(h)))),-acosd(double(uty(P(h)))),acosd(double(utz(P(h))))]);
            grid on;
            legend('Trajectory','Raduis of curvature curve','Velocity','Acceleration','Position','Raduis Of Curvature');
            hold off;
            % absolute velocity, normal acceleration, tangetial acceleration, position and raduis of cuevature
            figure(3);
            set(gcf,'Color','w')
            plot3(X,Y,Z,'blue','LineWidth',2); % curve
            hold on;
            xlim auto; ylim auto; zlim auto;
            vectarrow([0,0,0],double([x(P(h)),y(P(h)),z(P(h))]),2,'black'); % position
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+vx(P(h)),y(P(h))+vy(P(h)),z(P(h))+vz(P(h))]),2,[0.25,0.25,0.5]) % velocity
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+an(P(h))*unx(P(h)),y(P(h))+an(P(h))*uny(P(h)),z(P(h))+an(P(h))*unz(P(h))]),2,[0.7,0.2,0.5]); % normal acceleration
            vectarrow([x(P(h)),y(P(h)),z(P(h))],[x(P(h))+at(P(h))*utx(P(h)),y(P(h))+at(P(h))*uty(P(h)),z(P(h))+at(P(h))*utz(P(h))],2,'red'); % tangential acceleration
            vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
            vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
            vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
            view([-acosd(double(utx(P(h)))),-acosd(double(uty(P(h)))),acosd(double(utz(P(h))))]);
            grid on;
            title('Point Mass Moving on Trajectroy','Fontsize',18);
            xlabel('X','Fontsize',20);
            ylabel('Y','Fontsize',20);
            zlabel('Z','Fontsize',20);
            legend('Trajectory','Position','Velocity','Normal Acceleration','Tangential Acceleration');
            hold off;
            % Velocity Position Plane and Velocity Raduis Of Cuevature Plane
            figure(4);
            set(gcf,'Color','w')
            plot3(X,Y,Z,'blue','LineWidth',2); % curve
            hold on;
            xlim auto; ylim auto; zlim auto;
            % velocity position plane
            pointA = double([x(P(h)),y(P(h)),z(P(h))]);
            pointB = double([x(P(h))+vx(P(h)),y(P(h))+vy(P(h)),z(P(h))+vz(P(h))]);
            pointC = [0,0,0];
            plane1 = double(cross(pointA-pointB, pointA-pointC));
            points=double([pointA' pointB' pointC']); % using the data given in the question
            fill3(points(1,:),points(2,:),points(3,:),'r')
            alpha(0.5)
            % velocity raduis of cuevature plane
            pointA1 = double([x(P(h)),y(P(h)),z(P(h))]);
            pointB1 = double([x(P(h))+vx(P(h)),y(P(h))+vy(P(h)),z(P(h))+vz(P(h))]);
            pointC1 = double([x(P(h))+double(c(P(h))*unx(P(h))),y(P(h))+double(c(P(h))*uny(P(h))),z(P(h))+double(c(P(h))*unz(P(h)))]);
            plane11 = double(cross(pointA1-pointB1, pointA1-pointC1));
            points1=double([pointA1' pointB1' pointC1']); % using the data given in the question
            fill3(points1(1,:),points1(2,:),points1(3,:),'g')
            alpha(0.5)
            vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
            vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
            vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
            view([-acosd(double(utx(P(h)))),-acosd(double(uty(P(h)))),acosd(double(utz(P(h))))]);
            grid on;
            title('Point Mass Moving on Trajectroy','Fontsize',18);
            xlabel('X','Fontsize',20);
            ylabel('Y','Fontsize',20);
            zlabel('Z','Fontsize',20);
            legend('Trajectory','Velocity Position Plane','Velocity Raduis Of Cuevature Plane');
            hold off;
        end
    elseif (n > 0 && n < T(end) && m > 1.5 && m < 2)
        S=steps-n/atan((t_final-t_initial)/steps); % no. of steps
        P=linspace(n,t_final,S);
        for h=1:length(P)
            % absolute velocity, absolute acceleration, position and raduis of cuevature
            figure(5);
            set(gcf,'Color','w')
            view([-acosd(double(utx(P(h)))),-acosd(double(uty(P(h)))),acosd(double(utz(P(h))))]);
            plot3(X,Y,Z,'blue','LineWidth',2); % curve
            hold on;
            plot3(X+C.*Unx,Y+C.*Uny,Z+C.*Unz,'color',[0.7,0.25,0],'LineWidth',2); % raduis of curvature curve
            hold on;
            xlim auto; ylim auto; zlim auto;
            title('Point Mass Moving on Trajectroy','Fontsize',18);
            xlabel('X','Fontsize',20);
            ylabel('Y','Fontsize',20);
            zlabel('Z','Fontsize',20);
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+vx(P(h)),y(P(h))+vy(P(h)),z(P(h))+vz(P(h))]),2,[0.6,0.4,0]); % velocity
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+ax(P(h)),y(P(h))+ay(P(h)),z(P(h))+az(P(h))]),2,[0.4,0.7,0]); % acceleration
            vectarrow([0,0,0],double([x(P(h)),y(P(h)),z(P(h))]),2,'black'); % position
            vectarrow(double([x(P(h))+c(P(h))*unx(P(h)),y(P(h))+c(P(h))*uny(P(h)),z(P(h))+c(P(h))*unz(P(h))]),double([x(P(h)),y(P(h)),z(P(h))]),2,'yellow'); % raduis of curvature
            % absolute velocity, normal acceleration, tangetial acceleration, position and raduis of cuevature
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+an(P(h))*unx(P(h)),y(P(h))+an(P(h))*uny(P(h)),z(P(h))+an(P(h))*unz(P(h))]),2,[1,0.7,0]); % normal acceleration
            vectarrow(double([x(P(h)),y(P(h)),z(P(h))]),double([x(P(h))+at(P(h))*utx(P(h)),y(P(h))+at(P(h))*uty(P(h)),z(P(h))+at(P(h))*utz(P(h))]),2,'cyan'); % tangential acceleration
            % Velocity Position Plane and Velocity Raduis Of Cuevature Plane
            % velocity position plane
            pointA = double([x(P(h)),y(P(h)),z(P(h))]);
            pointB = double([x(P(h))+vx(P(h)),y(P(h))+vy(P(h)),z(P(h))+vz(P(h))]);
            pointC = [0,0,0];
            plane1 = double(cross(pointA-pointB, pointA-pointC));
            points=[pointA' pointB' pointC']; % using the data given in the question
            fill3(points(1,:),points(2,:),points(3,:),'r')
            alpha(0.5)
            % velocity raduis of cuevature plane
            pointA1 = double([x(P(h)),y(P(h)),z(P(h))]);
            pointB1 = double([x(P(h))+vx(P(h)),y(P(h))+vy(P(h)),z(P(h))+vz(P(h))]);
            pointC1 = double([x(P(h))+double(c(P(h))*unx(P(h))),y(P(h))+double(c(P(h))*uny(P(h))),z(P(h))+double(c(P(h))*unz(P(h)))]);
            plane11 = double(cross(pointA1-pointB1, pointA1-pointC1));
            points1=[pointA1' pointB1' pointC1']; % using the data given in the question
            fill3(points1(1,:),points1(2,:),points1(3,:),'blue')
            alpha(0.5)
            vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
            vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
            vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
            grid on;
            legend('Trajectory','Raduis of curvature curve','Velocity','Acceleration','Position','Raduis Of Curvature','Normal Acceleration','Tangential Acceleration','Velocity Position Plane','Velocity Raduis Of Cuevature Plane','Initial Axes');
            hold off;
            pause(1/hz);  
        end
    elseif (n > 0 && n < T(end) && m > 1 && m < 1.5)
        % absolute velocity, absolute acceleration, position and raduis of cuevature
        figure(5);
        view([-acosd(double(utx(n))),-acosd(double(uty(n))),acosd(double(utz(n)))]);
        set(gcf,'Color','w')
        plot3(X,Y,Z,'blue','LineWidth',2); % curve
        hold on;
        plot3(X+C.*Unx,Y+C.*Uny,Z+C.*Unz,'color',[0.7,0.25,0],'LineWidth',2); % raduis of curvature curve
        hold on;
        xlim auto; ylim auto; zlim auto;
        title('Point Mass Moving on Trajectroy','Fontsize',18);
        xlabel('X','Fontsize',20);
        ylabel('Y','Fontsize',20);
        zlabel('Z','Fontsize',20);
        vectarrow([double(x(n)),double(y(n)),double(z(n))],[double(x(n))+double(vx(n)),double(y(n))+double(vy(n)),double(z(n))+double(vz(n))],2,[0.6,0.4,0]); % velocity
        vectarrow(double([x(n),y(n),z(n)]),double([x(n)+ax(n),y(n)+ay(n),z(n)+az(n)]),2,[0.4,0.7,0]); % acceleration
        vectarrow([0,0,0],double([x(n),y(n),z(n)]),2,'black'); % position
        vectarrow(double([x(n)+c(n)*unx(n),y(n)+c(n)*uny(n),z(n)+c(n)*unz(n)]),double([x(n),y(n),z(n)]),2,'yellow'); % raduis of curvature
        % absolute velocity, normal acceleration, tangetial acceleration, position and raduis of cuevature
        vectarrow(double([x(n),y(n),z(n)]),double([x(n)+an(n)*unx(n),y(n)+an(n)*uny(n),z(n)+an(n)*unz(n)]),2,[1,0.7,0]); % normal acceleration
        vectarrow(double([x(n),y(n),z(n)]),double([x(n)+at(n)*utx(n),y(n)+at(n)*uty(n),z(n)+at(n)*utz(n)]),2,'cyan'); % tangential acceleration
        % Velocity Position Plane and Velocity Raduis Of Cuevature Plane
        % velocity position plane
        pointA = double([x(n),y(n),z(n)]);
        pointB = double([x(n)+vx(n),y(n)+vy(n),z(n)+vz(n)]);
        pointC = [0,0,0];
        plane1 = double(cross(pointA-pointB, pointA-pointC));
        points=[pointA' pointB' pointC']; % using the data given in the question
        fill3(points(1,:),points(2,:),points(3,:),'r')
        alpha(0.5)
        % velocity raduis of cuevature plane
        pointA1 = double([x(n),y(n),z(n)]);
        pointB1 = double([x(n)+vx(n),y(n)+vy(n),z(n)+vz(n)]);
        pointC1 = double([x(n)+double(c(n)*unx(n)),y(n)+double(c(n)*uny(n)),z(n)+double(c(n)*unz(n))]);
        plane11 = double(cross(pointA1-pointB1, pointA1-pointC1));
        points1=[pointA1' pointB1' pointC1']; % using the data given in the question
        fill3(points1(1,:),points1(2,:),points1(3,:),'blue')
        alpha(0.5)
        vectarrow([0,0,0],[0,0,max(double(Z))],2,'magenta');
        vectarrow([0,0,0],[0,max(double(Y)),0],2,'magenta');
        vectarrow([0,0,0],[max(double(X)),0,0],2,'magenta');
        grid on;
        legend('Trajectory','Raduis of curvature curve','Velocity','Acceleration','Position','Raduis Of Curvature','Normal Acceleration','Tangential Acceleration','Velocity Position Plane','Velocity Raduis Of Cuevature Plane','Initial Axes');
        hold off;        
    elseif (n > 0 && n < T(end) && m > 2 && m < 2.5)
        on=0;
    end
end
close all;

