% Defines rate function for potassium activation

function an = alphaN(v)
    an = (0.01 * (v + 55)) / (1 - exp(-(v + 55) / 10));
end

