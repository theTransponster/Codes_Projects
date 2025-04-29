
function Res = BF(p, u, t)                                                     % Función BS (Basis Functions) para generar las Shape Functions
    l = length(t);                                                              % Longitud del knot-vector
    k =  l-p-1;                                                                % Se calcula la cantidad de shape functions
    Ns1 = zeros(length(u),k);                                                  % Prelocaliza matriz Ns1 temporal 
    Res = Ns1;                                                                  % Prelocaliza matriz Res
    for j = 0:p                                                                % Ciclo para recorrer Shape functions desde grado 0 
        if j == 0                                                              % Si es grado 0
            for i = 1:k                                                        % Recorre las columnas para cada Shape function
                if i < k                                                       % Si es una columna diferente a la ultima 
                    Ns1(u >= t(i) & u < t(i+1),i) = 1;                         % No se considera el knot vector derecho
                else                                                           % Else
                    Ns1(u >= t(i) & u <= t(i+1),i) = 1;                        % Se considera el knot vector derecho
                end                                                            % Termina IF
            end                                                                % Termina FOR
            Res = Ns1 ;                                                         % Se guarda la matriz temporal Ns1 en Res
        else                                                                   % Else (si es grado mayor a 0)
            for i = 1:k                                                        % Se recorren las columnas de cada shape function
                utemp = u(u >= t(i) & u < t(i+j+1));                           % Se obtiene un vector u temporal
                anom = utemp-t(i);                                             % Se calcula el vector numerador de la variable a
                adenom = t(i+j)-t(i);                                          % Se calcula el vector denominador de la variable a

                bnom = t(i+j+1)-utemp;                                         % Se calcula el vector numerador de la variable b
                bdenom = t(i+j+1)-t(i+1);                                      % Se calcula el vector denominador de la variable b

                anom(adenom == 0) = 0;                                         % Si existe un cero en el denominador, el numerador es 0 
                adenom(adenom == 0) = 1;                                       % Si existe un cero en el denominador, el denominador es 1 

                bnom(bdenom == 0) = 0;                                         % Si existe un cero en el denominador, el numerador es 0 
                bdenom(bdenom == 0) = 1;                                       % Si existe un cero en el denominador, el denominador es 1 

                a = anom./adenom;                                              % Se calcula el vector a con el numerador y denominador
                b = bnom./bdenom;                                              % Se calcula el vector b con el numerador y denominador

                left  = a.*Ns1(u >= t(i) & u < t(i+j+1),i) ;                   % Se calcula la parte izquierda de la recursion Cox-de Boor
                
                if i < k                                                       % Si es una columna diferente a la ultima 
                    right = b.*Ns1(u >= t(i) & u < t(i+j+1),i+1) ;              % No se considera el knot vector derecho
                else                                                           % Else
                    right = 0     ;                                              % Se considera el knot vector derecho
                end
                
                Res(u >= t(i) & u < t(i+j+1),i) = left + right  ;                % Se calculan las shape functions en y se guardan en Res
            end                                                                % Termina FOR
            Ns1 = Res   ;                                                        % Se guarda la matriz temporal Ns1 en Res
        end                                                                    % Termina IF
    end                                                                        % Termina FOR
                                                                 % Termina función BF