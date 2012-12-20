function result=spherraytrace(Rcr,D,Ls,Ld,px,rec_pix_num)
%% parameters initialization
% primary parameters
% Rcr=25;
% D=6;
% Ls=50;
% Ld=1;
% Xdet=(0.5);
% rec_pix_num=1000;

%spherraytrace(Rcr,D,Ls,Ld,px,rec_pix_num)
%Rcr - радиус кривизны зеркала (25 см)
%D - диаметр зеркала (6 см)
%Ls - расстояние от зеркала до источника (50 см)
%Ld - расстояние от зеркала до детектора (2 см)
%px - координата в сигнале вдоль пятна (центр сигнала соответствует 0 см)
%rec_pix_num - разрешение реконструкции

%все длины в сантиметрах

%x0=-1.0;

% secondary parameters
%y0=-sqrt(0.25*D^2-x0.^2);
sa=D/(2*Rcr); %sin(alpha)
ca=sqrt(1-sa^2); %cos(beta)
rayfield=zeros(rec_pix_num);

% hold on;axis equal;
% xx=-0.5*D:0.01:0.5*D;
% plot(xx,-sqrt(0.25*D^2-xx.^2),'r');
% plot(xx,sqrt(0.25*D^2-xx.^2),'r');

%for j=1:length(Xdet)
%    px=Xdet(j);
    %% trace calculation
    Rs=[0 -Ls*ca-0.5*D -Rcr*ca+Ls*sa];
    Rd=[px Ld*ca+0.5*D -Rcr*ca+Ld*sa];
    Npl=cross(Rs,Rd);
    
    %% trace drawing
    nop=rec_pix_num; %number of points in the arc
    step=D/(nop-1);
    trY=-0.5*D:step:0.5*D;
    %trX=trY.*0;
    %trX(2)=L*ca*sg;
    for i=1:nop;
        y=trY(i);
        %equation coefficients
        a=Npl(1)^2+Npl(3)^2;
        b=y*Npl(1)*Npl(2);
        c=y^2*(Npl(2)^2+Npl(3)^2)-(Rcr*Npl(3))^2;
        x=(-b+sign(px)*sqrt(b^2-a*c))/a;
        x_=x+0.5*D;
        pn1=floor(x_/step)+1;
        ost=x_/step-(pn1-1);
        m1=abs(0.5-ost);
        m2=1-m1;
        if ost<0.5
            pn2=pn1-1;
            rayfield(pn1,i)=m2;
            rayfield(pn2,i)=m1;
        elseif ost>0.5
            pn2=pn1+1;
            rayfield(pn1,i)=m2;
            rayfield(pn2,i)=m1;
        else
            rayfield(pn1,i)=1;
        end
        %     trX(i)=x;
        %     if x^2+y^2<=0.25*D^2+0.001*D
        %         trX(i+2)=x;
        %     else
        %         trX(i+2)=trX(i+1)+(trX(i+1)-trX(i));
        %     end
    end
%    plot(trX,trY);
%end
result=rayfield;
