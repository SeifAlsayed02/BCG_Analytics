function states = BodyMovementDetection(BCG_raw)
    N = size(BCG_raw, 2); 
    states = repmat("rest", N, 1);
    S = zeros(N, 1); 

    for i = 1:N
        S(i) = std(BCG_raw(:, i));
    end

    MAD = mean(abs(S - mean(S)));
    AVE = mean(S);
    MED = median(S);

    for j = 1:N
        if S(j) > 4 * MAD
            states(j) = "motion";
            % 标记前两个状态
            if j > 2
                states(j-2:j-1) = "motion";
            elseif j > 1
                states(j-1) = "motion";
            end
            % 标记后三个状态
            if j < N-2
                states(j+1:j+3) = "motion";
            elseif j < N-1
                states(j+1:j+2) = "motion";
            elseif j < N
                states(j+1) = "motion";
            end
        elseif S(j) < MED * 0.08
            states(j) = "bedempty";
        end
    end
end
