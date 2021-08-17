//Para calculo de BV/TV
SetFactory("Built-in");
 
Merge "bone_3dslicer.vtk";
Surface Loop(100) = {1};//Hueso
hueso = newreg;
Volume(hueso) = {100};
Mesh.CharacteristicLengthMax = 0.05;

Merge "volumen_muestra.msh";
Plugin(NewView).Run;
Plugin(ModifyComponents).Expression0 = "1";
Plugin(ModifyComponents).Run;
Plugin(Integrate).Dimension = 3;
Plugin(Integrate).Run;
