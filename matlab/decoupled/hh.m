% Runs the Hodgkin Huxley model simulation

function y = hh(inter, ic, n, pa, pb, pulse)
    % Time grid setup
    a = inter(1);
    b = inter(2);
    h = (b-a)/n;

    % Initialize solution arrays
    y(1,:) = ic;
    t(1) = a;

    % Main integration loop
    for i = 1:n
        t(i+1) = t(i) + h;
        y(i+1,:) = rk4step(t(i), y(i,:), h, pa, pb, pulse);
    end

    % Plot results
    subplot(3,1,1);
    plot([a pa pa pb pb b], [0 0 pulse pulse 0 0]);
    grid; axis([0 100 0 2*pulse]);
    ylabel('input pulse');
    
    subplot(3,1,2);
    plot(t, y(:,1)); grid; axis([0 100 -100 100]);
    ylabel('voltage (mV)');
    
    subplot(3,1,3);
    plot(t, y(:,2), t, y(:,3), t, y(:,4)); grid; axis([0 100 0 1]);
    ylabel('gating variables');
    legend('m', 'h', 'n');
    xlabel('time (msec)');
end

