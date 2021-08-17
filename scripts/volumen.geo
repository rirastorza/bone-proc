//Gmsh calculo de volumen de malla

Merge "volumen_muestra.msh";
Plugin(NewView).Run;
Plugin(ModifyComponents).Expression0 = "1";
Plugin(ModifyComponents).Run;
Plugin(Integrate).Dimension = 3;
Plugin(Integrate).Run;


// Calculada: Step 0: integral = 14.60829195337311
