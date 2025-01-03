% Defines one runge kutta 4th order method step

function y = rk4step(t, w, h, pa, pb, pulse)
   
    s1 = ydot(t, w, pa, pb, pulse);
    s2 = ydot(t + h/2, w + h * s1/2, pa, pb, pulse);
    s3 = ydot(t + h/2, w + h * s2/2, pa, pb, pulse);
    s4 = ydot(t + h, w + h * s3, pa, pb, pulse);
    y = w + h * (s1 + 2*s2 + 2*s3 + s4) / 6;

end