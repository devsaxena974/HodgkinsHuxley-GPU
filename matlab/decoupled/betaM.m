% Defines beta_m for sodium activation

function bm = betaM(v)
    bm = 4 * exp(-(v + 65) / 18);
end