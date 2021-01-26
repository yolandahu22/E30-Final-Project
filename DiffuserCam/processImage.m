%% 
k=1
for i=35:-1:21
    path=string(i)+'.tif'
    x = imread(path', 'tif');
    psf(:,:,k)=x;
    k=k+1
end

save('testpsf.mat','psf')

%%
a = 1:600;
X = repelem(a,800*15);

b = 1:800;
Y = repmat(b,1,600*15);

c=1:15;
Z=repelem(c,480000);

%%
C=zeros(1,600*800*15);
z=1

for k=1:15
    for i=1:600
        for j=1:800
            C(z)=xhat_out(i,j,k);
            z=z+1;
        end
    end
end
%%
C=C*(256/max(C));
S=uint8(C)+1;
C=max(C)-C;
scatter3(X,Y,Z,1,C,'filled')
colormap(gray)

%% 
xhat_new=max(max(max(xhat_out)))-xhat_out;
count=0
max_val = max(max(max(xhat_out)))
[Nx, Ny,Nz] = size(xhat_new)
for k=1:Nz
     for i=150:(Nx-150)
        for j=200:(Ny-200)
            if (xhat_new(i,j,k)<max_val-0.2)
                count=count+1;
            end
        end
    end
end

%%
X=zeros(1,count);
Y=zeros(1,count);
Z=zeros(1,count);
C=zeros(1,count);
p=1;
for k=1:Nz
     for i=150:(Nx-150)
        for j=200:(Ny-200)
            if (xhat_new(i,j,k)<max_val-0.2)
                X(p)=i;
                Y(p)=j;
                Z(p)=k;
                C(p)=xhat_new(i,j,k);
                p=p+1;
            end
        end
    end
end

%%
scatter3(X,Y,Z,15,C,'filled')
colormap(copper)
