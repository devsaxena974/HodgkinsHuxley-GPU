% Defines the rate function for sodiumn inactivation

function bh = betaH(v)
    bh = 1 / (1 + exp(-(v + 35) / 10));
end