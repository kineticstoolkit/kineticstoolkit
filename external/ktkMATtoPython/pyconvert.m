    function pyres = pyconvert(ml)
        if isa(ml, 'struct')
            theFields = fields(ml);
            pyres = struct();
            for iField = 1:length(theFields)
                theField = theFields{iField};
                pyres.(theField) = pyconvert(ml.(theField));
            end
            
        elseif isa(ml, 'timeseries') || isa(ml, 'tsdata.event') || isa(ml, 'tsdata.timemetadata') || isa(ml, 'tsdata.datametadata')
            
            pyres = struct('type',class(ml));
            theProperties = properties(ml);
            for iProperty = 1:length(theProperties)
                theProperty = theProperties{iProperty};
                pyres.(theProperty) = pyconvert(ml.(theProperty));
            end
            
        elseif iscell(ml)
            
            pyres = struct();
            for i = 1:length(ml)
                pyres.(['cell' num2str(i)]) = ml{i};
            end
            
        elseif isa(ml, 'double') || isa(ml, 'char') || isa(ml, 'logical')
            pyres = ml;
            
        else
            try
                pyres = string(ml);
            catch
                pyres = NaN;
            end
        end
    end
    