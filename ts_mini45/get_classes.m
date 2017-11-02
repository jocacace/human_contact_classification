function [c0, c1, c2, c3] = get_classes( train )
    
    c0ids = find( train(:, 3) == 0 );
    c0 = train( min(c0ids):max(c0ids), : );
    
    c1ids = find( train(:, 3) == 1 );
    c1 = train( min(c1ids):max(c1ids), : );
    
    c2ids = find( train(:, 3) == 2 );
    c2 = train( min(c2ids):max(c2ids), : );
    
    c3ids = find( train(:, 3) == 3 );
    c3 = train( min(c3ids):max(c3ids), : );

end