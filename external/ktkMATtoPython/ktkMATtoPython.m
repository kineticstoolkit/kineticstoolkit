function ktkMATtoPython(sourceFile, destFile)

% Reads a MAT file, converts timeseries and structures of timeseries to
% simple structures to be read in Python, then saves as destFile.
%
% Author: Félix Chénier
% Date: July 15th, 2019

sourceData = load(sourceFile);

if ~isstruct(sourceData)
    error('The file contents should be read as a structure.');
end



contents = pyconvert(sourceData);
save(destFile, 'contents');

end
