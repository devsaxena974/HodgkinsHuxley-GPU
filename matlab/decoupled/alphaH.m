% Defines the rate function for sodium inactivation

function ah = alphaH(v)
    ah = 0.07 * exp(-(v + 65) / 20);
end