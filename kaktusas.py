import math
import add

def curve(t):
    x = t
    y = math.cos(t)
    return ([x,y])

vazonas = add.cylinder([0,0.6,0],[0,1.75,0],0.7,30,[255, 209, 220])
vazonoKrastas = add.cylinder([0,1.55,0],[0,1.76,0],0.8,30,[255, 209, 220])
zeme = add.cylinder([0,1.76,0],[0,1.77,0],0.6,30,[160,82,45])
kaktusas = add.cylinder([0,1.75,0],[0,3,0],0.3,30,[34,139,34])
kaktusoVirsus = add.sphere([0,3,0],0.3,15,[34,139,34])
kaktusasVazonas = add.layer()

kaktusoSakaX = add.cylinder([0,2.2,0],[0,3,0],0.2,30,[34,139,34])
kaktusoSakaX = add.layer()
kaktusoSakaX = add.rotateX(kaktusoSakaX, math.pi/2, [0,2.4,0])

kaktusoAlkune = add.sphere([0,2.4,0.6],0.2,15,[34,139,34])
kaktusoAlkunesV = add.sphere([0,2.8,0.6],0.2,15,[34,139,34])
kaktusoAlkune = add.layer()

kaktusoSakaY = add.cylinder([0,2.2,0],[0,2.6,0],0.2,30,[34,139,34])
kaktusoSakaY = add.layer()
kaktusoSakaY = add.move(kaktusoSakaY,[0,0.2,0.6])


akmenukas1 = add.sphere([0,1.75,0.45],0.08,15,[128,128,128])
akmenukas2 = add.sphere([0.06,1.75,0.45],0.065,15,[128,128,128])
akmenukas3 = add.sphere([-0.2,1.75,-0.40],0.08,15,[128,128,128])
akmenukai = add.layer()




stiklas = add.rectangle3D([-0.99,3.7,0],[0.1,6.5,4.79],[113,197,249])
palange = add.rectangle3D([0.1,0.5,0],[2.2,0.2,5.5],[255,255,255])
virsus = add.rectangle3D([0.2,7,0],[2.5,0.2,5.7],[255,255,255])
krastasV = add.rectangle3D([-0.8,7,0],[0.3,0.5,5.9],[255,255,255])
krastasP = add.rectangle3D([-0.8,0.5,0],[0.3,0.5,5.7],[255,255,255])
sonas1 = add.rectangle3D([-0.49,3.5,-2.3],[1,7,0.2],[255,255,255])
sonas2 = add.rectangle3D([-0.49,3.5,2.3],[1,7,0.2],[255,255,255])
centras = add.rectangle3D([-0.89,3.5,0],[0.2,7,0.2],[255,255,255])

karnizas = add.rectangle3D([1.5,7,0],[0.1,0.1,5.9],[128,128,128])
langas = add.layer()

uzuolaida1 = add.rectangle3D([2.4,3.1,2],[0.1,7.3,1],[255, 209, 220])
uzuolaida2 = add.rectangle3D([2.5,3.2,2.3],[0.1,7.2,1],[255, 209, 220])
uzuolaida3 = add.rectangle3D([2.4,3.1,-2],[0.1,7.3,1],[255, 209, 220])
uzuolaida4 = add.rectangle3D([2.5,3.2,-2.3],[0.1,7.2,1],[255, 209, 220])
uzuolaidos = add.layer()
uzuolaidos = add.rotateZ(uzuolaidos,math.pi/24,[0,0,0])

spindulys1 = add.rectangle3D([-0.98,5,-3],[0.1,0.1,1],[255, 255, 255])
spindulys2 = add.rectangle3D([-0.98,4.5,-3.5],[0.1,0.1,1],[255, 255, 255])
spindulys3 = add.rectangle3D([-0.98,2,-3],[0.1,0.1,1],[255, 255, 255])
spindulys4 = add.rectangle3D([-0.98,1.5,-3.5],[0.1,0.1,1],[255, 255, 255])
atspindys = add.layer()
atspindys = add.rotateX(atspindys,math.pi/4,[0,0,0])

add.spin3D([0,4,0],[0,0,0],curve,0,math.pi/2,5,5,[255,192,203])
ziedas = add.layer()
ziedas = add.zoom(ziedas,0.1)
ziedas = add.move(ziedas,[0,-0.05,0])

add.spin3D([0,4,0],[0,0,0],curve,0,math.pi/2,6,6,[255,192,203])
ziedas2 = add.layer()
ziedas2 = add.move(ziedas2,[0,1,0])
ziedas2 = add.zoom(ziedas2,0.14)





#kaktuso rankos
add.mesh(kaktusoSakaY)
add.mesh(kaktusoSakaX)
add.mesh(kaktusoAlkune)
kaktusoRanka = add.layer()
kaktusoRanka = add.move(kaktusoRanka,[0,0.1,0])
add.mesh(kaktusoRanka)
kaktusoRanka = add.rotateY(kaktusoRanka,math.pi,[0,0,0])
kaktusoRanka = add.move(kaktusoRanka,[0,-0.15,0])
add.mesh(kaktusoRanka)
#kaktuso rankos

#vazonas su pagr.
add.mesh(kaktusasVazonas)
#vazonas

#akmenukai
add.mesh(akmenukai)

add.mesh(langas)
add.mesh(uzuolaidos)
add.mesh(atspindys)

ziedas = add.rotateX(ziedas,math.pi/4,[0,3,0])
ziedas = add.rotateY(ziedas,math.pi/4,[0,3,0])
add.mesh(ziedas)
ziedas = add.rotateX(ziedas,math.pi/3,[0,3,0])
ziedas = add.rotateY(ziedas,math.pi/4,[0,3,0])
add.mesh(ziedas)

ziedas2 = add.rotateY(ziedas2,math.pi/4,[0,4,0])
ziedas2 = add.rotateX(ziedas2,-math.pi/4,[0,4,0])
ziedas2 = add.rotateZ(ziedas2,-math.pi/4,[0,4,0])
ziedas2 = add.move(ziedas2,[-0.09,-0.935,0.05])
add.mesh(ziedas2)


viskas = add.layer()
viskas = add.zoom(viskas,1)
add.mesh(viskas)

add.off("kaktusas.off")