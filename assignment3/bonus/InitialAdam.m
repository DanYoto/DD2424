function Adampara = InitialAdam(HyperParams)
layers = length(HyperParams.W);
SW = cell(1, layers); 
Sb = cell(1, layers); 
Sg = cell(1, layers); 
Sba = cell(1, layers); 
for i = 1:layers
    SW{i} = HyperParams.W{i}*0;
    Sb{i} = HyperParams.b{i}*0; 
    Sg{i} = HyperParams.gamma{i}*0; 
    Sba{i} = HyperParams.beta{i}*0; 
end
Adampara.mW = SW;
Adampara.mb = Sb;
Adampara.mbetas = Sba;
Adampara.mgammas = Sg;
Adampara.vW = SW;
Adampara.vb = Sb;
Adampara.vbetas = Sba;
Adampara.vgammas = Sg;
end