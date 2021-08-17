//Geometria

SetFactory("OpenCASCADE");
Cylinder(1) = {0, 0, -1.3, 0, 0, 2*1.3, 2.1, 2*Pi};

Delete { Volume{1}; } 


SetFactory("Built-in");
 
Merge "bone_meshlab.stl";

Surface Loop(100) = {1,2,3};//Cilindro
Surface Loop(200) = {4};//Hueso


marrow = newreg;
Volume(marrow) = {100,200}; // bone marrow
bone = newreg;
Volume(bone) = {200}; // bone

Physical Surface(10) = {2};//electrodo activo
Physical Surface(20) = {3};//electrodo pasivo

Physical Volume(100) = {marrow};//marrow
Physical Volume(200) = {bone};//hueso

Mesh.CharacteristicLengthMax = 0.1;
