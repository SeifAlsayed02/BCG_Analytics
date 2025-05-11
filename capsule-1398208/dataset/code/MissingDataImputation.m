function MissingDataImputation(datapath, newdatepath)
    [~, name, ~] = fileparts(datapath);
    if ~contains(name, 'RR')
        disp('File names that do not contain “RR” are not processed.');
        return;
    end

    if ~exist(newdatepath, 'dir')
        mkdir(newdatepath);
    end

    data = readtable(datapath, 'TextType', 'string');
    data.Properties.VariableNames = {'Timestamp','HR','RR'};

    timestamps = data.Timestamp;
    HR = data.HR;
    RR = data.RR;

    if ~isstring(timestamps)
        timestamps = string(timestamps);
    end

    newData = data(1, :);

    for i = 2:length(timestamps)
        t1 = datetime(timestamps(i-1), 'InputFormat', 'yyyy/MM/dd HH:mm:ss');
        t2 = datetime(timestamps(i), 'InputFormat', 'yyyy/MM/dd HH:mm:ss');
        timeDiff = seconds(t2 - t1);

        if timeDiff > 2
            numMissing = floor(timeDiff) - 1; 
            for j = 1:numMissing
                newTimestamp = t1 + seconds(j);
                newRow = {datestr(newTimestamp, 'yyyy/mm/dd HH:MM:ss'), 0, 0};
                newData = [newData; newRow];
            end
        end

        newData = [newData; data(i, :)];
    end

    [~, name, ext] = fileparts(datapath);
    newFileName = fullfile(newdatepath, [name, ext]);
    writetable(newData, newFileName);
end
