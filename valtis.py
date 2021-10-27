import add
import math

valtiesGalas = add.newface([[0,0,0],[0,-1.1,1.5],[0,0,3]],[139,69,19])
P = add.layer()
add.mesh(P)

valtieSiena1 = add.rectangle3D([0,0,2],[5,1,0.1],[139,69,19])
K = add.layer()
K = add.move(K,[0,-0.5,0])
K = add.rotateX(K,math.pi/4,[0,0,0])
add.mesh(K)

#valtieSiena2 = add.rectangle3D([0,0,0],[5,1,0.1],[139,69,19])
#valtiesGalas = add.newface([[0,0,0],[0,-1.1,1.5],[0,0,3]],[139,69,19])

#valtiesGalas = add.newface([[0,0,0],[0,1.1,0],[0,0,1.1]],[139,69,19])
#add.rotateX(add.rotateY(add.zoom(valtieSiena1,0.7),math.pi/4,[0,0,0]),-math.pi/4,[0,0,0])
axis = add.axes([0,0,0])


add.off("valtis.off")