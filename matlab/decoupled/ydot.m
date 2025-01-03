% Defines dynamics of the Hodkin-Huxley system

function z = ydot(t, w, pa, pb, pulse)
    % Parameters of the model
    Cm = 1;        % Membrane capacitance (μF/cm²)
    gNa = 120;     % Maximum sodium conductance (mS/cm²)
    gK = 36;       % Maximum potassium conductance (mS/cm²)
    gL = 0.3;      % Leak conductance (mS/cm²)
    ENa = 50;      % Sodium reversal potential (mV)
    EK = -77;      % Potassium reversal potential (mV)
    EL = -54.4;    % Leak reversal potential (mV)

    % For now, let's assume the input current is given
    %  by a square pulse
    T = (pa + pb)/2;
    len = pb - pa;
    Iext = pulse * (1 - sign(abs(t-T) - len/2)) / 2;

    % Get the current state variables
    v = w(1);
    m = w(2);
    h = w(3);
    n = w(4);

    % Calculate time constants and steady state values
    am = alphaM(v);
    bm = betaM(v);
    ah = alphaH(v);
    bh = betaH(v);
    an = alphaN(v);
    bn = betaN(v);

    tau_m = 1/(am + bm);     % typically 0.1-0.5 ms
    tau_h = 1/(ah + bh);     % typically 1-2 ms
    tau_n = 1/(an + bn);     % typically 1-2 ms

    m_inf = am/(am + bm);
    h_inf = ah/(ah + bh);
    n_inf = an/(an + bn);

    % Initialize the output vector
    z = zeros(1, 4);

    % voltage equation
    z(1) = (Iext - gNa*m^3*h*(v-ENa) - gK*n^4*(v-EK) - gL*(v-EL))/Cm;

    % gating variable equations
    z(2) = (m_inf - m)/tau_m;    % dm/dt
    z(3) = (h_inf - h)/tau_h;    % dh/dt
    z(4) = (n_inf - n)/tau_n;    % dn/dt
end


