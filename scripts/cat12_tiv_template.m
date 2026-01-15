% CAT12 TIV Calculation Template
% This script calculates Total Intracranial Volume (TIV) from CAT12 XML files
% Placeholders $XMLS and $OUT_FILE are replaced at runtime

% List of XML files
xml_files = {
$XMLS
};

% Calculate TIV for each file
n_files = length(xml_files);
tiv_values = zeros(n_files, 1);

for i = 1:n_files
    try
        tiv_values(i) = cat_vol_TIV(xml_files{i});
    catch
        warning('Failed to compute TIV for: %s', xml_files{i});
        tiv_values(i) = NaN;
    end
end

% Write results to output file
fid = fopen('$OUT_FILE', 'w');
for i = 1:n_files
    fprintf(fid, '%.6f\n', tiv_values(i));
end
fclose(fid);

fprintf('TIV calculation complete. Results saved to: $OUT_FILE\n');
