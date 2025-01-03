% Defines rate function for potassium activation gate

function bn = betaN(v)
    bn = 0.125 * exp(-(v + 65) / 80);
end