
function [BCGRRI_normal] = OutlierReplacement(BCGRRI_raw, window_size, outlier_threshold_upper, outlier_threshold_lower, num_of_cycles)
    
    if any(BCGRRI_raw)
        % window_size = 300; 
        z_threshold = 3; 
        % outlier_threshold1 = 2; 
        % outlier_threshold2 = 0.5; 
        
        BCGRRI_iteration = BCGRRI_raw;
        BCGRRI_normal_table = zeros(length(BCGRRI_raw), num_of_cycles); 
    
        for iter = 1:num_of_cycles
            for i = 1:length(BCGRRI_iteration)
                window_start = max(1, i - floor(window_size / 2));
                window_end = min(length(BCGRRI_iteration), i + floor(window_size / 2));
                window_data = BCGRRI_iteration(window_start:window_end);
    
                local_median = median(window_data);
                local_std = std(window_data);
    
                if BCGRRI_iteration(i) >= outlier_threshold_upper || BCGRRI_iteration(i) <= outlier_threshold_lower || ...
                        abs(BCGRRI_iteration(i) - local_median) > z_threshold * local_std
                    BCGRRI_iteration(i) = local_median;
                end
            end
            BCGRRI_normal_table(:, iter) = BCGRRI_iteration;
        end

%         figure;
%         hold on;
%         plot(timeRRI,BCGRRI_raw, 'k', 'DisplayName', '原始RRI');
%         plot(timeRRI,BCGRRI_normal_table(:, 1), 'r', 'DisplayName', '第一次迭代');
%         plot(timeRRI,BCGRRI_normal_table(:, 2), 'g', 'DisplayName', '第二次迭代');
%         plot(BCGRRI_normal_table(:, 3), 'b', 'DisplayName', '第三次迭代');
%         hold off;
%         legend;
%         title('【BCGRRI】每次异常值处理结果图','FontName','simsun');
%         xlabel('Index');
%         ylabel('RRI Value');
    end
    BCGRRI_normal=BCGRRI_normal_table(:,num_of_cycles);