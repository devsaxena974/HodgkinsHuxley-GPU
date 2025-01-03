% Defines alpha_m for sodium channel activation

function a_m = alphaM(v)
    a_m = (0.1 * (v + 40)) / (1 - exp(-(v+40) * 0.1));
end
