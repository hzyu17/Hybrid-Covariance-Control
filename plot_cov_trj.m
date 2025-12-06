function output = plot_cov_trj(zk, Sk, nt, m0, Sig0, tf, x_trj_samples)
%%
n  = 2;

% The first half time period
n1 = nt;
t  = linspace(0,tf,nt);
n3=120;
theta=linspace(0,2*pi,n3);
circlepara=[cos(theta);sin(theta)];
n4=500;
tellips=zeros(n,n3,n4);
tellips(:,:,1)=3*sqrtm(Sig0)*circlepara+m0*ones(1,n3);
for i=1:n4-1
    i_ell = ceil(1+(n1-1)/(n4-1)*i);
    
    tellips(:,:,i+1)=3*sqrtm(Sk(:,:,i_ell))*circlepara+zk(:,i_ell)*ones(1,n3);
end
X=zeros(n4,n3);Y=zeros(n4,n3);Z=zeros(n4,n3);
for i=1:n4-1
    i_ell = ceil(1+(n1-1)/(n4-1)*i);
    if i_ell == 1231
        i_event = i;
    end
    X(i,:)=t(i_ell)*ones(1,n3);
    Y(i,:)=tellips(1,:,i);
    Z(i,:)=tellips(2,:,i);
end
cstring='grbcmk';
%cstring='cccccc';
figure(1),hold on;
% grid("minor")
%mesh(X,Y,Z);

surf(X(1:i_event, :),Y(1:i_event, :),Z(1:i_event, :),'FaceColor','blue','EdgeColor','none');
surf(X(i_event+1:end-1, :),Y(i_event+1:end-1, :),Z(i_event+1:end-1, :),'FaceColor','blue','EdgeColor','none');
alpha(0.1);
view(-60, 20)

n_samples = size(x_trj_samples, 1);
for j = 1:n_samples
    plot3(t,x_trj_samples(j, :, 1),x_trj_samples(j, :, 2),cstring(mod(j,6)+1),'LineWidth',2);
end

% for i=1:n4
%     plot3(t(1+(n1-1)/(n4-1)*(i-1))*ones(1,n3),tellips(1,:,i),tellips(2,:,i));
% end
xlabel('Time $t$','Interpreter','latex'),ylabel('Position $z$','Interpreter','latex');
zlabel('Velocity $\dot z$','Interpreter','latex');
set(gca,'fontsize',16);
output = 1;
grid minor
end
